"""
Lightweight FastAPI backend for the multi-style AI interviewer.
Exposes a WebSocket endpoint for the interview loop and a simple health check.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import re
import time
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from io import BytesIO
from fastapi import FastAPI, File, Form, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlmodel import SQLModel
from tempfile import NamedTemporaryFile
from collections import OrderedDict

import httpx
from app.db import get_session, init_db
from app.models import AnswerRecord, CheckInRecord, SessionRecord, TelemetryRecord
from faster_whisper import WhisperModel
from TTS.api import TTS
import soundfile as sf
import logging
import traceback


class InterviewerStyle(str, Enum):
    SUPPORTIVE = "supportive"
    NEUTRAL = "neutral"
    COLD = "cold"


class SessionState:
    """Per-connection state to keep the loop simple and testable."""

    def __init__(
        self,
        style: InterviewerStyle = InterviewerStyle.NEUTRAL,
        group: str = "treatment",
        consented: bool = False,
        accent: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> None:
        self.session_id = str(uuid.uuid4())
        self.style = style
        self.group = group
        self.consented = consented
        self.accent = accent
        self.notes = notes
        self.turn = 0
        self.last_question: Optional[str] = None
        self.history: List[Tuple[str, str]] = []  # recent (question, answer) pairs
        self.awaiting_followup: bool = False


QUESTION_BANK: Dict[InterviewerStyle, List[str]] = {
    InterviewerStyle.SUPPORTIVE: [
        "Tell me about a project you loved working on and why.",
        "What is a strength you’re proud of, and how do you use it on teams?",
    ],
    InterviewerStyle.NEUTRAL: [
        "Walk me through a challenging problem you recently solved.",
        "Describe a time you disagreed with a teammate. What happened?",
    ],
    InterviewerStyle.COLD: [
        "Why should we trust you with high-stakes work?",
        "Explain a recent mistake. How did you recover? Be concise.",
    ],
}

FOLLOW_UPS: Dict[InterviewerStyle, List[str]] = {
    InterviewerStyle.SUPPORTIVE: [
        "Nice—can you share a detail that shows your impact?",
        "How did that experience shape how you collaborate now?",
    ],
    InterviewerStyle.NEUTRAL: [
        "What was your exact role, and what did you deliver?",
        "What would you do differently if you faced this again?",
    ],
    InterviewerStyle.COLD: [
        "Your answer felt light on specifics. Give numbers.",
        "Time is short—summarize the critical decision you made.",
    ],
}


def pick_question(style: InterviewerStyle, turn: int) -> str:
    items = QUESTION_BANK.get(style) or QUESTION_BANK[InterviewerStyle.NEUTRAL]
    return items[turn % len(items)]


def generate_coaching(style: InterviewerStyle, answer: str) -> List[Dict[str, str]]:
    """Tiny heuristic coach: varies tone per style and reacts to length/pauses."""
    length = len(answer.strip())
    has_pause = "..." in answer or "  " in answer
    fillers = sum(answer.lower().count(f) for f in [" um", " uh", " like "])

    base_tips: List[Dict[str, str]] = []

    if length < 80:
        base_tips.append(
            {
                "summary": "Add one concrete detail",
                "detail": "Include a metric or decision so the interviewer can see impact.",
            }
        )
    if has_pause:
        base_tips.append(
            {
                "summary": "Smooth your pacing",
                "detail": "Finish sentences before pausing; keep eye contact for the last word.",
            }
        )
    if fillers > 1:
        base_tips.append(
            {
                "summary": "Trim fillers",
                "detail": "Take a breath before answering to reduce “uh/um” clusters.",
            }
        )
    if not base_tips:
        base_tips.append(
            {
                "summary": "Good structure",
                "detail": "Keep using concise, past-tense statements to anchor the story.",
            }
        )

    # Style-specific framing for novelty and tone.
    tone_prefix = {
        InterviewerStyle.SUPPORTIVE: "Encouraging: ",
        InterviewerStyle.NEUTRAL: "",
        InterviewerStyle.COLD: "Direct: ",
    }[style]
    return [{**tip, "detail": f"{tone_prefix}{tip['detail']}"} for tip in base_tips[:2]]


def _extract_json_block(text: str) -> Optional[Dict[str, Any]]:
    """Tolerant JSON extraction so we survive code fences or preambles."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return None
    return None


def parse_coaching_response(text: str) -> Optional[Tuple[Optional[str], List[Dict[str, str]]]]:
    data = _extract_json_block(text)
    if not data:
        return None
    follow_up = (data.get("follow_up") or data.get("followup") or data.get("followUp") or "").strip()
    tips_payload = data.get("tips") or []
    tips: List[Dict[str, str]] = []
    if isinstance(tips_payload, list):
        for item in tips_payload:
            if not isinstance(item, dict):
                continue
            summary = str(item.get("summary") or "").strip()
            detail = str(item.get("detail") or item.get("text") or "").strip()
            if summary and detail:
                tips.append({"summary": summary, "detail": detail})
            if len(tips) >= 2:
                break
    if not follow_up and not tips:
        return None
    return follow_up or None, tips


def parse_question_response(text: str) -> Optional[str]:
    data = _extract_json_block(text)
    question: Optional[str] = None
    if data:
        question = data.get("question") or data.get("prompt") or data.get("text")
    if isinstance(question, list):
        question = " ".join(str(part) for part in question)
    if not isinstance(question, str) or not question:
        # Fall back to interpreting the raw text as the question (helps if the LLM ignored JSON instructions).
        question = (text or "").strip().splitlines()[0] if text else None
    if not question:
        return None
    cleaned = question.strip().strip('"').strip()
    if cleaned and not cleaned.endswith("?"):
        cleaned = f"{cleaned}?"
    return cleaned or None


async def llm_generate_coaching(
    style: InterviewerStyle, question: str, answer: str, turn: int
) -> Optional[Tuple[Optional[str], List[Dict[str, str]]]]:
    NVIDIA_API_KEY = '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'
    if not NVIDIA_API_KEY:
        LOG.warning("NVIDIA_API_KEY missing; coaching fallback engaged (style=%s turn=%s)", style, turn)
        return None
    system_prompt = (
        "You are an interview coach speaking as the interviewer. "
        "Respond with JSON only: {\"follow_up\":\"string\",\"tips\":[{\"summary\":\"string\",\"detail\":\"string\"},...]}. "
        "Keep follow_up one sentence, pointed and natural. Return 2 concise, actionable tips that reference delivery "
        "(pacing, confidence, specificity). Tone varies by style: supportive=warm & encouraging; neutral=direct & calm; "
        "cold=pressuring and blunt. Avoid markdown/code fences."
    )
    user_prompt = (
        f"Style: {style.value}\nTurn: {turn}\nQuestion: {question}\nUser answer: {answer}\n"
        "Return JSON only."
    )
    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    payload = {
        "model": NVIDIA_LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": 240,
        "temperature": 0.8,
        "top_p": 1.0,
        "stream": False,
    }
    try:
        async with httpx.AsyncClient(timeout=NVIDIA_LLM_TIMEOUT) as client:
            LOG.info(
                "Calling NVIDIA LLM (coaching): style=%s turn=%s question_len=%s answer_len=%s",
                style,
                turn,
                len(question),
                len(answer),
            )
            resp = await client.post(NVIDIA_LLM_URL, headers=headers, json=payload)
    except Exception as exc:  # pragma: no cover - network/runtime safety
        LOG.warning("NVIDIA LLM coaching request failed: %s", exc)
        return None

    if resp.status_code != 200:
        LOG.warning("NVIDIA LLM responded with %s (coaching): %s", resp.status_code, resp.text[:200])
        return None

    try:
        data = resp.json()
        choices = data.get("choices") or []
        content = choices[0].get("message", {}).get("content", "").strip() if choices else ""
    except Exception:
        content = ""
    if not content:
        LOG.warning("NVIDIA LLM coaching returned empty content")
        return None
    parsed = parse_coaching_response(content)
    if not parsed:
        LOG.warning("NVIDIA LLM coaching parse failed; raw content: %s", content[:200])
    return parsed


async def llm_generate_question(
    style: InterviewerStyle, turn: int, previous_question: Optional[str] = None, history: Optional[List[Tuple[str, str]]] = None
) -> Optional[str]:
    NVIDIA_API_KEY = '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'
    if not NVIDIA_API_KEY:
        LOG.warning("NVIDIA_API_KEY missing; question fallback engaged (style=%s turn=%s)", style, turn)
        return None
    system_prompt = (
        "You are an interviewer generating the next question. Respond with JSON only: {\"question\":\"string\"}. "
        "Tone varies by style: supportive=warm/encouraging, neutral=calm/direct, cold=pressuring/blunt. "
        "Keep it concise, natural, and behaviorally specific. Avoid code fences or commentary."
    )
    recent_pairs = history[-3:] if history else []
    history_block = "\n".join([f"Q: {q}\nA: {a}" for q, a in recent_pairs]) or "None yet"
    user_prompt = (
        f"Style: {style.value}\n"
        f"Turn index (0-based): {turn}\n"
        f"Previous question: {previous_question or 'None'}\n"
        f"Recent Q/A (most recent last):\n{history_block}\n"
        "Give the next single interview question in JSON only. Ask something that logically follows the last answer; avoid repeats."
    )
    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    payload = {
        "model": NVIDIA_LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": 80,
        "temperature": 0.85,
        "top_p": 1.0,
        "stream": False,
    }
    try:
        async with httpx.AsyncClient(timeout=NVIDIA_LLM_TIMEOUT) as client:
            LOG.info(
                "Calling NVIDIA LLM (question): style=%s turn=%s prev_len=%s history_pairs=%s",
                style,
                turn,
                len(previous_question or ""),
                len(recent_pairs),
            )
            resp = await client.post(NVIDIA_LLM_URL, headers=headers, json=payload)
    except Exception as exc:  # pragma: no cover - network/runtime safety
        LOG.warning("NVIDIA LLM question request failed: %s", exc)
        return None

    if resp.status_code != 200:
        LOG.warning("NVIDIA LLM question responded with %s: %s", resp.status_code, resp.text[:200])
        return None

    try:
        data = resp.json()
        choices = data.get("choices") or []
        content = choices[0].get("message", {}).get("content", "").strip() if choices else ""
    except Exception:
        content = ""
    if not content:
        LOG.warning("NVIDIA LLM question returned empty content")
        return None
    parsed = parse_question_response(content)
    if not parsed:
        LOG.warning("NVIDIA LLM question parse failed; raw content: %s", content[:200])
    return parsed


app = FastAPI(title="AI Interview Trainer", version="0.1.0")
whisper_model: Optional[WhisperModel] = None
tts_model: Optional[TTS] = None
TTS_SPEAKER = os.getenv("TTS_SPEAKER")
TTS_LANGUAGE = os.getenv("TTS_LANGUAGE", "en")
DEFAULT_TTS_SPEAKER: Optional[str] = None
FALLBACK_TTS_SPEAKER = os.getenv("TTS_FALLBACK_SPEAKER", "Claribel Dervla")
TTS_CACHE_MAX = 32  # simple in-memory LRU for duplicate TTS requests
TTS_CACHE: "OrderedDict[tuple, bytes]" = OrderedDict()
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
NVIDIA_LLM_MODEL = os.getenv("NVIDIA_LLM_MODEL", "meta/llama-4-maverick-17b-128e-instruct")
NVIDIA_LLM_URL = os.getenv("NVIDIA_LLM_URL", "https://integrate.api.nvidia.com/v1/chat/completions")
NVIDIA_LLM_TIMEOUT = float(os.getenv("NVIDIA_LLM_TIMEOUT", "12"))
LOG = logging.getLogger("interview")


@app.on_event("startup")
async def on_startup() -> None:
    await init_db()
    model_size = os.getenv("WHISPER_MODEL", "medium")
    device = os.getenv("WHISPER_DEVICE", "cpu")
    compute_type = os.getenv("WHISPER_COMPUTE_TYPE") or ("float16" if device not in ("cpu", "auto-cpu") else "int8")
    global whisper_model
    try:
        whisper_model = WhisperModel(model_size, device=device, compute_type=compute_type)
        print(f"[whisper] loaded {model_size} on {device} ({compute_type})")
    except Exception as exc:  # pragma: no cover - defensive
        # Fall back to None so the app still boots even if model load fails.
        whisper_model = None
        print(f"[whisper] failed to load model {model_size}: {exc}")

    tts_name = os.getenv("TTS_MODEL", "tts_models/multilingual/multi-dataset/xtts_v2")
    tts_device = os.getenv("TTS_DEVICE", device)
    global tts_model, DEFAULT_TTS_SPEAKER, TTS_SPEAKER
    try:
        tts_model = TTS(model_name=tts_name).to(tts_device)
        DEFAULT_TTS_SPEAKER = None
        # Try several ways to discover a default speaker so /tts works out of the box.
        if hasattr(tts_model, "speakers") and tts_model.speakers:
            DEFAULT_TTS_SPEAKER = tts_model.speakers[0]
        elif getattr(tts_model, "speaker_manager", None) and getattr(tts_model.speaker_manager, "speaker_names", None):
            DEFAULT_TTS_SPEAKER = tts_model.speaker_manager.speaker_names[0]
        # Last-resort fallback so /tts never hard-fails on speaker selection.
        if DEFAULT_TTS_SPEAKER is None:
            DEFAULT_TTS_SPEAKER = FALLBACK_TTS_SPEAKER
        if TTS_SPEAKER is None and DEFAULT_TTS_SPEAKER:
            TTS_SPEAKER = DEFAULT_TTS_SPEAKER
        print(f"[tts] loaded {tts_name} on {tts_device}; default speaker={DEFAULT_TTS_SPEAKER}")
    except Exception as exc:
        tts_model = None
        print(f"[tts] failed to load model {tts_name}: {exc}")

# CORS for local dev; adjust allowed origins for prod if needed.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


class TtsRequest(BaseModel):
    text: str
    style: Optional[str] = None
    speaker: Optional[str] = None
    language: Optional[str] = None


@app.post("/tts")
async def tts_endpoint(payload: TtsRequest) -> Response:
    """Neural TTS via Coqui XTTS; returns WAV bytes."""
    if tts_model is None:
        return Response(content=json.dumps({"error": "tts_not_loaded"}), media_type="application/json", status_code=500)

    text = payload.text.strip()
    if not text:
        return Response(content=json.dumps({"error": "empty_text"}), media_type="application/json", status_code=400)

    style = payload.style or "neutral"
    speed = {"supportive": 1.06, "neutral": 1.0, "cold": 0.94}.get(style, 1.0)
    speaker = payload.speaker or TTS_SPEAKER or DEFAULT_TTS_SPEAKER or FALLBACK_TTS_SPEAKER
    if speaker is None:
        candidates: List[str] = []
        try:
            if hasattr(tts_model, "speakers") and tts_model.speakers:
                candidates = list(tts_model.speakers)
            elif hasattr(tts_model, "speaker_manager") and hasattr(tts_model.speaker_manager, "speaker_names"):
                candidates = list(tts_model.speaker_manager.speaker_names)
        except Exception:
            candidates = []
        if candidates:
            speaker = candidates[0]
        else:
            # Still nothing? use the hard-coded fallback to avoid hard 400s.
            speaker = FALLBACK_TTS_SPEAKER
    cache_key = (text, style, speaker, payload.language or TTS_LANGUAGE, speed)
    if cache_key in TTS_CACHE:
        # Move to end to keep LRU ordering.
        audio_bytes = TTS_CACHE.pop(cache_key)
        TTS_CACHE[cache_key] = audio_bytes
        return Response(content=audio_bytes, media_type="audio/wav")
    try:
        wav = tts_model.tts(
            text=text,
            speaker=speaker,
            language=payload.language or TTS_LANGUAGE,
            speed=speed,
        )
        sample_rate = getattr(tts_model.synthesizer, "output_sample_rate", 24000)
        buf = BytesIO()
        sf.write(buf, wav, sample_rate, format="WAV")
        audio_bytes = buf.getvalue()
        TTS_CACHE[cache_key] = audio_bytes
        if len(TTS_CACHE) > TTS_CACHE_MAX:
            TTS_CACHE.popitem(last=False)
        return Response(content=audio_bytes, media_type="audio/wav")
    except Exception as exc:  # pragma: no cover - runtime safeguard
        logging.error("TTS synthesis failed: %s\n%s", exc, traceback.format_exc())
        return Response(content=json.dumps({"error": f"tts_failed: {exc}"}), media_type="application/json", status_code=500)


@app.post("/stt")
async def transcribe_audio(
    file: UploadFile = File(...),
    session_id: Optional[str] = Form(default=None, alias="sessionId"),
    language: Optional[str] = Form(default=None),
) -> Dict[str, Any]:
    """Speech-to-text via local Whisper (faster-whisper)."""
    if whisper_model is None:
        return {"error": "whisper_not_loaded"}

    started = time.perf_counter()
    suffix = os.path.splitext(file.filename or "audio.webm")[1] or ".webm"
    payload = await file.read()
    if not payload:
        LOG.warning("STT received empty payload (session=%s)", session_id)
        return {"error": "empty_audio"}
    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(payload)
        tmp_path = tmp.name

    transcript = ""
    info_payload: Dict[str, Any] = {}
    try:
        segments, info = whisper_model.transcribe(
            tmp_path,
            beam_size=4,
            language=language or "en",
            vad_filter=True,
            condition_on_previous_text=False,
        )
        texts: List[str] = []
        for seg in segments:
            seg_text = seg.text.strip()
            if seg_text:
                texts.append(seg_text)
        transcript = " ".join(texts).strip()
        info_payload = {"duration": info.duration, "language": info.language, "num_segments": len(texts)}
    except Exception as exc:
        LOG.warning("STT failed: %s", exc)
        return {"error": f"stt_failed: {exc}"}
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    latency_ms = round((time.perf_counter() - started) * 1000, 2)

    if session_id:
        async with get_session() as session:
            group = None
            session_row = await session.get(SessionRecord, session_id)
            if session_row:
                group = session_row.group_name
            session.add(
                TelemetryRecord(
                    session_id=session_id,
                    event_type="stt",
                    latency_ms=latency_ms,
                    group_name=group,
                    payload=json.dumps(
                        {"language": language or info_payload.get("language"), "duration": info_payload.get("duration")}
                    ),
                )
            )
            await session.commit()

    return {
        "transcript": transcript,
        "latency_ms": latency_ms,
        **info_payload,
    }


async def send_question(
    ws: WebSocket,
    state: SessionState,
    override_question: Optional[str] = None,
    source: Optional[str] = None,
) -> None:
    question_source = source or "llm"
    if override_question:
        question = override_question
    else:
        question = await llm_generate_question(state.style, state.turn, state.last_question, state.history)
        if not question:
            question_source = "fallback"
            question = pick_question(state.style, state.turn)
            LOG.info("Question fallback: style=%s turn=%s", state.style, state.turn)

    state.last_question = question
    await ws.send_json(
        {"type": "question", "turn": state.turn, "question": question, "style": state.style, "source": question_source}
    )
    if question_source == "follow_up":
        await ws.send_json(
            {
                "type": "interviewer_message",
                "turn": state.turn,
                "style": state.style,
                "message": question,
            }
        )


async def send_reaction(ws: WebSocket, state: SessionState, answer: str, question: str, turn: int) -> Tuple[str, List[Dict[str, str]]]:
    follow_up: Optional[str] = None
    tips: Optional[List[Dict[str, str]]] = None

    llm_result = await llm_generate_coaching(state.style, question, answer, turn)
    if llm_result:
        follow_up, tips = llm_result

    if not follow_up:
        options = FOLLOW_UPS.get(state.style, FOLLOW_UPS[InterviewerStyle.NEUTRAL])
        follow_up = random.choice(options)
        LOG.info("Follow-up fallback: style=%s turn=%s", state.style, turn)
    if not tips:
        tips = generate_coaching(state.style, answer)
        LOG.info("Tips fallback: style=%s turn=%s", state.style, turn)

    return follow_up or "", tips


async def handle_message(ws: WebSocket, state: SessionState, payload: Dict[str, Any]) -> None:
    msg_type = payload.get("type")
    if msg_type == "start_session":
        requested_style = payload.get("style")
        group = payload.get("group") or state.group
        state.turn = 0
        state.last_question = None
        state.history = []
        state.awaiting_followup = False
        state.consented = bool(payload.get("consent"))
        state.accent = payload.get("accent")
        state.notes = payload.get("notes")
        if requested_style and requested_style in InterviewerStyle._value2member_map_:
            state.style = InterviewerStyle(requested_style)
        state.group = group
        async with get_session() as session:
            session.add(
                SessionRecord(
                    id=state.session_id,
                    style=state.style.value,
                    group_name=group,
                    consented=state.consented,
                    accent=state.accent,
                    notes=state.notes,
                )
            )
            await session.commit()
        await ws.send_json(
            {
                "type": "session_started",
                "session_id": state.session_id,
                "style": state.style,
                "turn": state.turn,
                "group": group,
                "consent": state.consented,
            }
        )
        await send_question(ws, state)
        return

    if msg_type == "switch_style":
        new_style = payload.get("style")
        if new_style and new_style in InterviewerStyle._value2member_map_:
            state.style = InterviewerStyle(new_style)
            await ws.send_json({"type": "style_switched", "style": state.style})
            await send_question(ws, state)
        return

    if msg_type == "user_answer":
        answer = payload.get("answer", "")
        asked_question = state.last_question or pick_question(state.style, state.turn)
        state.turn += 1
        turn_label = state.turn  # maintain existing turn numbering for the UI/DB
        metrics = payload.get("metrics") or {}
        async with get_session() as session:
            session.add(
                AnswerRecord(
                    session_id=state.session_id,
                    turn=turn_label,
                    answer=answer,
                    style=state.style.value,
                    group_name=state.group,
                    speaking_rate=metrics.get("speakingRate"),
                    pause_ratio=metrics.get("pauseRatio"),
                    gaze=metrics.get("gaze"),
                    fillers=metrics.get("fillers"),
                )
            )
            await session.commit()
        # keep a small rolling history to guide the next question
        state.history.append((asked_question, answer))
        if len(state.history) > 4:
            state.history = state.history[-4:]

        follow_up, tips = await send_reaction(ws, state, answer, asked_question, turn_label)
        if tips:
            await ws.send_json({"type": "tips", "turn": turn_label, "items": tips})

        if state.awaiting_followup:
            # This answer was for a follow-up; resume normal questioning.
            state.awaiting_followup = False
            await send_question(ws, state)
        else:
            # This answer was for the main prompt; if we have a follow-up, ask it and wait for the answer before moving on.
            if follow_up:
                state.awaiting_followup = True
                await send_question(ws, state, override_question=follow_up, source="follow_up")
            else:
                await send_question(ws, state)
        return

    if msg_type == "ping":
        await ws.send_json({"type": "pong"})
        return

    if msg_type == "checkin":
        async with get_session() as session:
            session.add(
                CheckInRecord(
                    session_id=state.session_id,
                    group_name=payload.get("group") or state.group,
                    confidence=int(payload.get("confidence", 0)),
                    stress=int(payload.get("stress", 0)),
                )
            )
            await session.commit()
        await ws.send_json({"type": "checkin_logged"})
        return

    if msg_type == "telemetry":
        async with get_session() as session:
            session.add(
                TelemetryRecord(
                    session_id=state.session_id,
                    event_type=payload.get("event") or "unknown",
                    latency_ms=payload.get("latencyMs"),
                    group_name=state.group,
                    payload=json.dumps(payload.get("data") or {}),
                )
            )
            await session.commit()
        return

    await ws.send_json({"type": "error", "message": f"Unrecognized message type: {msg_type}"})


@app.websocket("/ws/interview")
async def interview_socket(ws: WebSocket) -> None:
    await ws.accept()
    state = SessionState()
    await ws.send_json({"type": "session_ready", "session_id": state.session_id, "style": state.style})

    try:
        while True:
            raw = await ws.receive_text()
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                await ws.send_json({"type": "error", "message": "Payload must be JSON"})
                continue

            await handle_message(ws, state, payload)
            await asyncio.sleep(0)  # yield control
    except WebSocketDisconnect:
        return


class CheckInPayload(BaseModel):
    session_id: str = Field(..., alias="sessionId")
    group: Optional[str] = None
    confidence: int
    stress: int


@app.post("/checkins")
async def log_checkin(payload: CheckInPayload) -> Dict[str, str]:
    async with get_session() as session:
        session.add(
            CheckInRecord(
                session_id=payload.session_id,
                group_name=payload.group,
                confidence=payload.confidence,
                stress=payload.stress,
            )
        )
        await session.commit()
    return {"status": "ok"}


@app.get("/export/session/{session_id}")
async def export_session(session_id: str) -> Dict[str, Any]:
    async with get_session() as session:
        session_row = await session.get(SessionRecord, session_id)
        answers = (
            await session.exec(select(AnswerRecord).where(AnswerRecord.session_id == session_id))
        ).all()
        checkins = (
            await session.exec(select(CheckInRecord).where(CheckInRecord.session_id == session_id))
        ).all()
        telemetry = (
            await session.exec(select(TelemetryRecord).where(TelemetryRecord.session_id == session_id))
        ).all()

        if not session_row:
            return {"error": "not_found"}

        return {
            "session": session_row.model_dump(),
            "answers": [a.model_dump() for a in answers],
            "checkins": [c.model_dump() for c in checkins],
            "telemetry": [t.model_dump() for t in telemetry],
        }


@app.get("/metrics/summary")
async def metrics_summary() -> Dict[str, Any]:
    async with get_session() as session:
        stats: Dict[str, Dict[str, Any]] = {}
        for group in ["control", "treatment"]:
            answers = (
                await session.exec(select(AnswerRecord).where(AnswerRecord.group_name == group))
            ).all()
            checkins = (
                await session.exec(select(CheckInRecord).where(CheckInRecord.group_name == group))
            ).all()
            telemetry = (
                await session.exec(
                    select(TelemetryRecord).where(
                        TelemetryRecord.event_type == "latency", TelemetryRecord.group_name == group
                    )
                )
            ).all()
            stt_events = (
                await session.exec(
                    select(TelemetryRecord).where(
                        TelemetryRecord.event_type == "stt", TelemetryRecord.group_name == group
                    )
                )
            ).all()

            def mean(values: List[Optional[float]]) -> Optional[float]:
                vals = [v for v in values if v is not None]
                return sum(vals) / len(vals) if vals else None

            stats[group] = {
                "n_answers": len(answers),
                "n_checkins": len(checkins),
                "n_latency": len(telemetry),
                "n_stt": len(stt_events),
                "avg_speaking_rate": mean([a.speaking_rate for a in answers]),
                "avg_pause_ratio": mean([a.pause_ratio for a in answers]),
                "avg_gaze": mean([a.gaze for a in answers]),
                "avg_fillers": mean([float(a.fillers) if a.fillers is not None else None for a in answers]),
                "avg_confidence": mean([float(c.confidence) for c in checkins]),
                "avg_stress": mean([float(c.stress) for c in checkins]),
                "avg_latency_ms": mean([t.latency_ms for t in telemetry]),
                "avg_stt_latency_ms": mean([t.latency_ms for t in stt_events]),
            }

        def delta(key: str) -> Optional[float]:
            t = stats["treatment"].get(key)
            c = stats["control"].get(key)
            if t is None or c is None:
                return None
            return t - c

        return {
            "groups": stats,
            "delta": {
                "speaking_rate": delta("avg_speaking_rate"),
                "pause_ratio": delta("avg_pause_ratio"),
                "gaze": delta("avg_gaze"),
                "fillers": delta("avg_fillers"),
                "confidence": delta("avg_confidence"),
                "stress": delta("avg_stress"),
            },
        }
