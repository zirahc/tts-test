"""Streamlit bench for a text-to-speech API."""

from __future__ import annotations

import os
import time
from datetime import datetime
from typing import Any

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from tts_client import ApiError, TtsClient

PROVIDER = "elevenlabs"
TERMINAL_STATUSES = {"completed", "failed", "error", "cancelled", "canceled"}
POLL_SECONDS = 1.0
POLL_TIMEOUT_SECONDS = 180
VOICE_PAGE_SIZE = 100
MAX_VOICE_PAGES = 100


def main() -> None:
    st.set_page_config(page_title="TTS API bench", layout="wide")
    st.title("TTS API bench")

    api_key, base_url = _settings()
    if not api_key or not base_url:
        st.error("Add TTS_API_KEY and TTS_API_BASE_URL to the .env file, then restart the app.")
        return

    data = _load_catalog(api_key, base_url)
    models_tab, voices_tab, bench_tab = st.tabs(["Models", "Voices", "Performance"])
    with models_tab:
        _render_models(data)
    with voices_tab:
        _render_voices(data)
    with bench_tab:
        _render_benchmark(api_key, base_url, data)


def _settings() -> tuple[str, str]:
    load_dotenv()
    api_key = os.getenv("TTS_API_KEY", "").strip()
    base_url = os.getenv("TTS_API_BASE_URL", "").strip().rstrip("/")
    return api_key, base_url


@st.cache_data(show_spinner=False)
def _load_catalog(api_key: str, base_url: str) -> dict[str, Any]:
    client = TtsClient(api_key, base_url)
    errors: list[str] = []
    models: list[dict[str, Any]] = []
    languages: dict[str, list[dict[str, str]]] = {}
    voices: list[dict[str, Any]] = []
    me: dict[str, Any] = {}

    try:
        me, _elapsed_ms = client.me()
    except ApiError as exc:
        errors.append(str(exc))

    try:
        rows, _elapsed_ms = client.models(PROVIDER)
        for row in rows:
            models.append({**row, "provider": PROVIDER})
    except ApiError as exc:
        errors.append(str(exc))
    try:
        langs, _elapsed_ms = client.languages(PROVIDER)
        languages[PROVIDER] = langs
    except ApiError as exc:
        errors.append(str(exc))
        languages[PROVIDER] = []

    voice_jobs = [
        ("Default voices", PROVIDER, "default", "/v1/default-voices", ("voices",), 0, None),
        ("Shared voices", PROVIDER, "shared", "/v1/shared-voices", ("voices",), 0, {"sort": "trending"}),
    ]
    for label, provider, source, path, keys, start, extra in voice_jobs:
        batch = client.collect_pages(
            path,
            label=label,
            list_keys=keys,
            start_page=start,
            max_pages=MAX_VOICE_PAGES,
            page_size=VOICE_PAGE_SIZE,
            extra=extra,
        )
        if batch.error:
            errors.append(f"{label}: {batch.error}")
        voices.extend(
            _normalize_voice(item, provider, source, batch.truncated, batch.reported_total)
            for item in batch.items
        )
        if batch.truncated:
            errors.append(f"{label}: stopped after {MAX_VOICE_PAGES} pages; more voices are available.")

    return {
        "base_url": base_url,
        "loaded_at": datetime.now().isoformat(timespec="seconds"),
        "me": me,
        "models": models,
        "languages": languages,
        "voices": voices,
        "errors": errors,
    }


def _normalize_voice(
    item: dict[str, Any],
    provider: str,
    source: str,
    truncated: bool,
    reported_total: int | None,
) -> dict[str, Any]:
    voice_id = str(item.get("voice_id") or item.get("id") or item.get("uniq_id") or "")
    tags = item.get("tag_list") or item.get("tags") or []
    if not isinstance(tags, list):
        tags = [str(tags)]
    language = item.get("language") or item.get("language_tag") or ""
    if not language and tags:
        language = ", ".join(str(tag) for tag in tags[:4])
    return {
        "provider": provider,
        "source": source,
        "voice_id": voice_id,
        "uniq_id": str(item.get("uniq_id") or ""),
        "name": str(item.get("name") or item.get("voice_name") or voice_id),
        "language": str(language),
        "gender": str(item.get("gender") or ""),
        "accent": str(item.get("accent") or ""),
        "description": str(item.get("description") or ""),
        "preview_url": str(
            item.get("preview_url") or item.get("sample_audio") or item.get("sample_audio_url") or ""
        ),
        "category": str(item.get("category") or ""),
        "list_capped": truncated,
        "reported_total": reported_total,
    }


def _model_row(item: dict[str, Any]) -> dict[str, Any]:
    languages = item.get("languages") or []
    if isinstance(languages, list):
        names = []
        for lang in languages:
            if isinstance(lang, dict):
                names.append(str(lang.get("language_id") or lang.get("name") or lang.get("code") or ""))
            else:
                names.append(str(lang))
        language_text = ", ".join(name for name in names if name)
    else:
        language_text = str(languages)
    ratio = item.get("credit_ratio")
    if ratio is None:
        rates = item.get("model_rates")
        if isinstance(rates, dict):
            ratio = rates.get("character_cost_multiplier")
    if ratio is None:
        ratio = item.get("token_cost_factor")
    return {
        "provider": item.get("provider", ""),
        "model_id": str(item.get("model_id") or item.get("id") or ""),
        "name": str(item.get("name") or item.get("model_id") or ""),
        "description": str(item.get("description") or ""),
        "tts": item.get("can_do_text_to_speech", True),
        "credit_ratio": ratio,
        "languages": language_text,
    }


def _render_models(catalog: dict[str, Any]) -> None:
    frame = pd.DataFrame([_model_row(item) for item in catalog["models"]])
    if not frame.empty and "provider" in frame:
        frame = frame[frame["provider"] == PROVIDER]
    st.metric("Models", 0 if frame.empty else len(frame))

    if frame.empty:
        st.info("No models returned.")
        return

    query = st.text_input("Filter models", placeholder="Name or model id")
    view = frame
    if query.strip():
        needle = query.strip().lower()
        mask = view.apply(lambda row: needle in " ".join(str(value).lower() for value in row), axis=1)
        view = view[mask]
    st.dataframe(view, hide_index=True, width="stretch")
    st.download_button(
        "Download models CSV",
        view.to_csv(index=False).encode(),
        "models.csv",
        "text/csv",
    )


def _render_voices(catalog: dict[str, Any]) -> None:
    frame = pd.DataFrame(catalog["voices"])
    if frame.empty:
        st.info("No voices returned.")
        return

    if "provider" in frame:
        frame = frame[frame["provider"] == PROVIDER]
    counts = frame.groupby("source").size()
    summary = " · ".join(f"{source}: {int(count)}" for source, count in counts.items())
    st.caption(f"{len(frame)} ElevenLabs voices. {summary}")

    sources = sorted(frame["source"].dropna().unique().tolist())
    chosen_sources = st.multiselect("Source", sources, default=sources)
    query = st.text_input("Filter voices", placeholder="Name, language, or voice id")
    view = frame[frame["source"].isin(chosen_sources)] if chosen_sources else frame.iloc[0:0]
    if query.strip():
        needle = query.strip().lower()
        haystack = (
            view["name"].fillna("")
            + " "
            + view["voice_id"].fillna("")
            + " "
            + view["uniq_id"].fillna("")
            + " "
            + view["language"].fillna("")
            + " "
            + view["gender"].fillna("")
            + " "
            + view["accent"].fillna("")
        ).str.lower()
        view = view[haystack.str.contains(needle, regex=False)]

    show = view.drop(columns=["list_capped", "reported_total"], errors="ignore")
    event = st.dataframe(
        show,
        hide_index=True,
        width="stretch",
        selection_mode="single-row",
        on_select="rerun",
        key="voice_table",
    )
    st.download_button(
        "Download voices CSV",
        show.to_csv(index=False).encode(),
        "voices.csv",
        "text/csv",
    )

    selected = _selected_row(event, view)
    if selected is None:
        st.caption("Select a row to preview it and send it to the performance tab.")
        return

    st.subheader(str(selected.get("name") or "Voice"))
    st.write(f"ElevenLabs · {selected['source']} · `{selected['voice_id']}`")
    if selected.get("description"):
        st.write(selected["description"])
    preview = str(selected.get("preview_url") or "")
    if preview:
        st.audio(preview)
    else:
        st.caption("This voice has no preview URL.")
    if st.button("Use this voice in the performance test"):
        st.session_state["bench_voice"] = {
            "provider": selected["provider"],
            "voice_id": selected["voice_id"],
            "name": selected["name"],
        }
        st.success("Voice selected. Open the Performance tab.")


def _render_benchmark(api_key: str, base_url: str, catalog: dict[str, Any]) -> None:
    st.write(
        "Submits one speech job and polls until it finishes. "
        f"Polling is every {POLL_SECONDS:.0f}s, so completion time is accurate to about that interval. "
        "Each run spends credits."
    )

    models = [_model_row(item) for item in catalog["models"]]
    voices = [row for row in catalog["voices"] if row.get("provider") == PROVIDER]
    _apply_voice_preset()

    provider = PROVIDER
    provider_models = [row for row in models if row["provider"] == provider and row["model_id"]]
    speakable = [row for row in provider_models if row.get("tts") is not False]
    model_choices = speakable or provider_models
    if not model_choices:
        st.info("No models returned.")
        return

    model = st.selectbox(
        "Model",
        model_choices,
        format_func=lambda row: f"{row['name']} ({row['model_id']})",
        key=f"bench_model_{provider}",
    )
    voice_choices = [row for row in voices if row["provider"] == provider and row["voice_id"]]
    search = st.text_input("Find a voice", key="voice_search")
    filtered = _filter_voices(voice_choices, search)
    labels = {_voice_label(row): row for row in filtered[:300]}
    if not labels:
        st.info("No voices match. Paste a voice id below.")
    elif len(filtered) > 300:
        st.caption(f"Showing 300 of {len(filtered)} matches. Narrow the search to see the rest.")

    options = list(labels)
    voice_label = st.selectbox(
        "Voice",
        options or ["—"],
        key=f"bench_voice_choice_{provider}_{search.strip().lower()}",
        disabled=not options,
    )
    picked = labels.get(voice_label)
    override = st.text_input(
        "Voice id override",
        placeholder="Leave blank to use the voice selected above",
        help="Sent in the speech request URL.",
    )
    voice_id = override.strip() or str((picked or {}).get("voice_id") or "")

    langs = catalog.get("languages", {}).get(provider) or []
    if langs:
        default_index = _default_language_index(langs)
        language = st.selectbox(
            "Language",
            langs,
            index=default_index,
            format_func=lambda row: row["name"] if row["name"] == row["code"] else f"{row['name']} ({row['code']})",
            key=f"bench_language_{provider}",
        )
        language_code = language["code"]
    else:
        language_code = st.text_input("Language code", value="en").strip()

    text = st.text_area(
        "Text",
        value="The quick brown fox jumps over the lazy dog.",
        height=100,
    )
    speed_col, stability_col, similarity_col = st.columns(3)
    speed = speed_col.slider("Speed", min_value=0.7, max_value=1.2, value=1.0, step=0.01)
    stability = stability_col.slider("Stability", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    similarity = similarity_col.slider("Similarity", min_value=0.0, max_value=1.0, value=0.75, step=0.01)
    repeats = st.number_input("Repeats", min_value=1, max_value=5, value=1)
    ratio = model.get("credit_ratio")
    estimate = len(text) * float(ratio) if ratio not in (None, "") else len(text)
    st.caption(f"{len(text)} characters. Rough credit estimate for one run: {estimate:g}.")

    if st.button("Run speech test", type="primary", disabled=not (voice_id and text.strip() and language_code)):
        client = TtsClient(api_key, base_url)
        runs = st.session_state.setdefault("runs", [])
        for index in range(int(repeats)):
            st.write(f"Run {index + 1} of {int(repeats)}")
            run = _run_once(
                client,
                provider=provider,
                model_id=model["model_id"],
                voice_id=voice_id.strip(),
                voice_name=(picked or {}).get("name") or voice_id.strip(),
                language_code=language_code,
                text=text.strip(),
                speed=float(speed),
                stability=float(stability),
                similarity=float(similarity),
            )
            _remember_audio(client, run)
            runs.append({key: value for key, value in run.items() if key not in {"audio_bytes", "audio_type"}})
            _show_run(run)

    last_audio = st.session_state.get("last_audio")
    if last_audio and last_audio.get("bytes"):
        st.caption(str(last_audio.get("label") or "Latest audio"))
        st.audio(last_audio["bytes"], format=last_audio.get("format") or "audio/mpeg")

    history = st.session_state.get("runs") or []
    if history:
        st.subheader("Results")
        history_frame = pd.DataFrame(history)
        chart = history_frame[["accept_s", "total_s"]].apply(pd.to_numeric, errors="coerce")
        st.bar_chart(chart)
        st.dataframe(history_frame, hide_index=True, width="stretch")
        st.download_button(
            "Download results CSV",
            history_frame.to_csv(index=False).encode(),
            "bench.csv",
            "text/csv",
        )


def _run_once(
    client: TtsClient,
    *,
    provider: str,
    model_id: str,
    voice_id: str,
    voice_name: str,
    language_code: str,
    text: str,
    speed: float,
    stability: float,
    similarity: float,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "text": text,
        "model_id": model_id,
        "provider": provider,
        "language_code": language_code,
        "voice_settings": _voice_settings(speed, stability, similarity),
    }
    progress = st.progress(0, text="Submitting")
    started = time.perf_counter()
    run: dict[str, Any] = {
        "at": datetime.now().isoformat(timespec="seconds"),
        "provider": provider,
        "model_id": model_id,
        "voice_id": voice_id,
        "voice": voice_name,
        "language_code": language_code,
        "speed": speed,
        "stability": stability,
        "similarity": similarity,
        "characters": len(text),
        "status": "error",
        "accept_s": None,
        "total_s": None,
        "polls": 0,
        "credits": None,
        "audio_url": "",
        "error": "",
    }
    try:
        body, accept_ms, http_status, content_type = client.submit_tts(voice_id, payload)
    except ApiError as exc:
        run["error"] = str(exc)
        run["total_s"] = round(time.perf_counter() - started, 2)
        progress.progress(100, text="Submit failed")
        return run

    run["accept_s"] = round(accept_ms / 1000, 3)
    if isinstance(body, (bytes, bytearray)):
        run["status"] = "completed"
        run["total_s"] = run["accept_s"]
        run["audio_bytes"] = bytes(body)
        run["audio_type"] = content_type
        progress.progress(100, text=f"Audio returned with HTTP {http_status}")
        return run

    task_id = ""
    if isinstance(body, dict):
        task_id = str(body.get("id") or body.get("task_id") or "")
    if not task_id:
        run["error"] = f"HTTP {http_status} did not include a task id: {body}"
        run["total_s"] = round(time.perf_counter() - started, 2)
        progress.progress(100, text="No task id")
        return run

    run["task_id"] = task_id
    progress.progress(8, text=f"Queued {task_id}")
    deadline = time.perf_counter() + POLL_TIMEOUT_SECONDS
    task: dict[str, Any] = {}
    while True:
        try:
            task, _elapsed = client.history_detail(task_id)
        except ApiError as exc:
            run["error"] = str(exc)
            break
        run["polls"] += 1
        status = str(task.get("status") or "")
        percent = task.get("progress")
        shown = int(percent) if isinstance(percent, (int, float)) else min(95, 8 + run["polls"] * 5)
        progress.progress(max(8, min(99, shown)), text=f"{status or 'working'} · poll {run['polls']}")
        if status.lower() in TERMINAL_STATUSES:
            run["status"] = status
            break
        if time.perf_counter() >= deadline:
            run["status"] = status or "timeout"
            run["error"] = f"No terminal status within {POLL_TIMEOUT_SECONDS}s"
            break
        time.sleep(POLL_SECONDS)

    run["total_s"] = round(time.perf_counter() - started, 2)
    run["credits"] = task.get("credits_deducted")
    run["characters_used"] = task.get("characters_used")
    error = task.get("error")
    if error and not run["error"]:
        run["error"] = str(error)
    audio_url = _audio_url(task)
    if audio_url:
        run["audio_url"] = audio_url
    if str(run["status"]).lower() == "completed":
        progress.progress(100, text=f"Completed in {run['total_s']}s")
    else:
        progress.progress(100, text=f"{run['status']} in {run['total_s']}s")
    return run


def _apply_voice_preset() -> None:
    preset = st.session_state.get("bench_voice") or {}
    preset_id = str(preset.get("voice_id") or "")
    if not preset_id or st.session_state.get("applied_preset") == preset_id:
        return
    st.session_state["voice_search"] = str(preset.get("name") or "")
    st.session_state["applied_preset"] = preset_id


def _remember_audio(client: TtsClient, run: dict[str, Any]) -> None:
    if run.get("audio_bytes"):
        st.session_state["last_audio"] = {
            "bytes": run["audio_bytes"],
            "format": _audio_format(str(run.get("audio_type") or "")),
            "label": f"{run.get('voice')} · {run.get('model_id')}",
        }
        return
    url = str(run.get("audio_url") or "")
    if not url or str(run.get("status") or "").lower() != "completed":
        return
    try:
        st.session_state["last_audio"] = {
            "bytes": client.fetch_audio(url),
            "format": "audio/mpeg",
            "label": f"{run.get('voice')} · {run.get('model_id')}",
        }
    except ApiError as exc:
        run["error"] = run.get("error") or f"Could not download audio: {exc}"


def _show_run(run: dict[str, Any]) -> None:
    cols = st.columns(4)
    cols[0].metric("Accept", _seconds(run.get("accept_s")))
    cols[1].metric("Total", _seconds(run.get("total_s")))
    cols[2].metric("Status", str(run.get("status") or ""))
    cols[3].metric("Credits", "—" if run.get("credits") is None else str(run.get("credits")))
    if run.get("error"):
        st.error(run["error"])


def _voice_settings(speed: float, stability: float, similarity: float) -> dict[str, Any]:
    return {
        "stability": stability,
        "similarity_boost": similarity,
        "speed": speed,
    }


def _audio_url(task: dict[str, Any]) -> str:
    result = task.get("result")
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        for key in ("audio_url", "url"):
            if result.get(key):
                return str(result[key])
    if task.get("audio_url"):
        return str(task["audio_url"])
    return ""


def _filter_voices(voices: list[dict[str, Any]], query: str) -> list[dict[str, Any]]:
    needle = query.strip().lower()
    if not needle:
        return voices
    matched = []
    for voice in voices:
        blob = " ".join(
            str(voice.get(key) or "")
            for key in ("name", "voice_id", "uniq_id", "language", "gender", "accent", "source")
        ).lower()
        if needle in blob:
            matched.append(voice)
    return matched


def _voice_label(voice: dict[str, Any]) -> str:
    return f"{voice['name']} · {voice['source']} · {voice['voice_id']}"


def _default_language_index(langs: list[dict[str, str]]) -> int:
    wanted = {"english", "en"}
    for index, lang in enumerate(langs):
        if lang["code"].lower() in wanted or lang["name"].lower() in wanted:
            return index
    return 0


def _selected_row(event: Any, frame: pd.DataFrame) -> dict[str, Any] | None:
    selection = getattr(event, "selection", None)
    rows = getattr(selection, "rows", None)
    if rows is None and isinstance(selection, dict):
        rows = selection.get("rows")
    if not rows:
        return None
    position = rows[0]
    if position >= len(frame):
        return None
    return frame.iloc[position].to_dict()


def _seconds(value: Any) -> str:
    if value is None:
        return "—"
    return f"{float(value):.2f} s"


def _audio_format(content_type: str) -> str:
    if "wav" in content_type:
        return "audio/wav"
    if "ogg" in content_type:
        return "audio/ogg"
    return "audio/mpeg"


if __name__ == "__main__":
    main()
