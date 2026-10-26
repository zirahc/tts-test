"""HTTP client for the speech API configured in the environment."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import quote

import requests


class ApiError(Exception):
    def __init__(self, method: str, path: str, status: int, body: str):
        self.method = method
        self.path = path
        self.status = status
        self.body = body
        super().__init__(f"{method} {path} failed ({status}): {body}")


@dataclass
class CallTiming:
    label: str
    method: str
    path: str
    status: int
    ms: float
    count: int = 0
    note: str = ""


@dataclass
class PageBatch:
    items: list[dict[str, Any]]
    truncated: bool
    reported_total: int | None
    timings: list[CallTiming] = field(default_factory=list)
    error: str = ""


class TtsClient:
    def __init__(self, api_key: str, base_url: str, timeout: float = 45):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(
            {
                "xi-api-key": api_key,
                "Accept": "application/json",
                "User-Agent": "TTSBench/1.0",
            }
        )

    def request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> tuple[requests.Response, float]:
        url = self.base_url + path
        started = time.perf_counter()
        try:
            response = self.session.request(
                method,
                url,
                params=params,
                json=json_body,
                timeout=timeout or self.timeout,
            )
        except requests.RequestException as exc:
            raise ApiError(method, path, 0, f"{exc.__class__.__name__}: {exc}") from exc
        elapsed_ms = (time.perf_counter() - started) * 1000
        return response, elapsed_ms

    def json(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
        timeout: float | None = None,
        ok_statuses: tuple[int, ...] = (200,),
    ) -> tuple[Any, float, int]:
        response, elapsed_ms = self.request(
            method, path, params=params, json_body=json_body, timeout=timeout
        )
        if response.status_code not in ok_statuses:
            raise ApiError(method, path, response.status_code, _short_body(response))
        if not response.content:
            return None, elapsed_ms, response.status_code
        try:
            return response.json(), elapsed_ms, response.status_code
        except ValueError as exc:
            raise ApiError(method, path, response.status_code, "Response was not JSON") from exc

    def get_json(self, path: str, params: dict[str, Any] | None = None) -> tuple[Any, float, int]:
        return self.json("GET", path, params=params)

    def me(self) -> tuple[dict[str, Any], float]:
        data, elapsed_ms, _status = self.get_json("/v1/auth/me")
        return data if isinstance(data, dict) else {}, elapsed_ms

    def models(self, provider: str | None = None) -> tuple[list[dict[str, Any]], float]:
        params = {"provider": provider} if provider else None
        data, elapsed_ms, _status = self.get_json("/v1/models", params)
        return _as_records(data, ("models", "data")), elapsed_ms

    def languages(self, provider: str | None = None) -> tuple[list[dict[str, str]], float]:
        params = {"provider": provider} if provider else None
        data, elapsed_ms, _status = self.get_json("/v1/languages", params)
        rows = _as_records(data, ("languages", "data"))
        cleaned: list[dict[str, str]] = []
        for row in rows:
            code = str(row.get("code") or row.get("language_id") or row.get("id") or "").strip()
            name = str(row.get("name") or code).strip()
            if code:
                cleaned.append({"code": code, "name": name})
        return cleaned, elapsed_ms

    def get_records(
        self,
        path: str,
        *,
        label: str,
        list_keys: tuple[str, ...],
        params: dict[str, Any] | None = None,
    ) -> PageBatch:
        try:
            data, elapsed_ms, status = self.get_json(path, params)
        except ApiError as exc:
            return PageBatch([], False, None, [], str(exc))
        items = _as_records(data, list_keys)
        reported = data.get("total") if isinstance(data, dict) and isinstance(data.get("total"), int) else None
        timing = CallTiming(label, "GET", path, status, elapsed_ms, len(items))
        return PageBatch(items, False, reported, [timing])

    def collect_pages(
        self,
        path: str,
        *,
        label: str,
        list_keys: tuple[str, ...],
        start_page: int,
        max_pages: int,
        page_size: int = 100,
        extra: dict[str, Any] | None = None,
    ) -> PageBatch:
        items: list[dict[str, Any]] = []
        seen: set[str] = set()
        timings: list[CallTiming] = []
        reported_total: int | None = None
        truncated = False
        page = start_page

        for index in range(max_pages):
            params: dict[str, Any] = {"page_size": page_size, "page": page}
            if extra:
                params.update({k: v for k, v in extra.items() if v not in (None, "")})
            try:
                data, elapsed_ms, status = self.get_json(path, params)
            except ApiError as exc:
                if items and exc.status in (400, 404, 422):
                    timings.append(
                        CallTiming(label, "GET", path, exc.status, 0, note="pagination stopped")
                    )
                    break
                return PageBatch([], False, reported_total, timings, str(exc))

            batch = _as_records(data, list_keys)
            if isinstance(data, dict) and isinstance(data.get("total"), int):
                reported_total = data["total"]
            fresh: list[dict[str, Any]] = []
            for item in batch:
                if not isinstance(item, dict):
                    continue
                identity = _voice_identity(item)
                if identity in seen:
                    continue
                seen.add(identity)
                fresh.append(item)
            items.extend(fresh)
            timings.append(CallTiming(label, "GET", f"{path}?page={page}", status, elapsed_ms, len(fresh)))

            has_more = data.get("has_more") if isinstance(data, dict) else None
            if not batch or not fresh or has_more is False:
                break
            if has_more is None and len(batch) < page_size:
                break
            if index == max_pages - 1:
                truncated = True
                break
            page += 1

        return PageBatch(items, truncated, reported_total, timings)

    def submit_tts(
        self,
        voice_id: str,
        payload: dict[str, Any],
    ) -> tuple[Any, float, int, str]:
        path = f"/v1/text-to-speech/{quote(str(voice_id), safe='')}"
        response, elapsed_ms = self.request("POST", path, json_body=payload, timeout=90)
        content_type = response.headers.get("content-type", "")
        if response.status_code not in (200, 202):
            raise ApiError("POST", path, response.status_code, _short_body(response))
        if "audio" in content_type and response.status_code == 200:
            return response.content, elapsed_ms, response.status_code, content_type
        try:
            body = response.json() if response.content else {}
        except ValueError as exc:
            raise ApiError("POST", path, response.status_code, "Response was not JSON") from exc
        return body, elapsed_ms, response.status_code, content_type

    def history_detail(self, task_id: str) -> tuple[dict[str, Any], float]:
        path = f"/v1/history/{quote(str(task_id), safe='')}"
        data, elapsed_ms, _status = self.get_json(path)
        return data if isinstance(data, dict) else {}, elapsed_ms

    def fetch_audio(self, url: str) -> bytes:
        if url.startswith("/"):
            url = self.base_url + url
        if url.startswith(self.base_url):
            response = self.session.get(url, timeout=60)
        else:
            response = requests.get(url, timeout=60)
        if response.status_code >= 400:
            raise ApiError("GET", url, response.status_code, _short_body(response))
        return response.content


def _short_body(response: requests.Response, limit: int = 400) -> str:
    text = (response.text or "").strip().replace("\n", " ")
    if len(text) > limit:
        return text[:limit] + "..."
    return text or response.reason


def _as_records(data: Any, keys: tuple[str, ...]) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]
    if isinstance(data, dict):
        for key in keys:
            value = data.get(key)
            if isinstance(value, list):
                return [row for row in value if isinstance(row, dict)]
        if any(k in data for k in ("model_id", "voice_id", "id", "code")):
            return [data]
    return []


def _voice_identity(item: dict[str, Any]) -> str:
    for key in ("voice_id", "id", "uniq_id"):
        value = item.get(key)
        if value not in (None, ""):
            return str(value)
    name = item.get("name") or item.get("voice_name") or ""
    return f"name:{name}"
