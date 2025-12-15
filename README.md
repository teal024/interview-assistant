# AI Interview Trainer (multi-style)

Prototype for the ergonomics-focused, multi-style AI interviewer described in `Ergonomics-Proposal.pdf`. It ships a Next.js UI with style switching, live WebRTC/WebAudio + MediaPipe gaze HUD, and a FastAPI WebSocket backend that delivers questions plus tone-aware coaching with persistence.

## Layout
- `frontend/` — Next.js (TypeScript) app with interview loop, live mic/cam telemetry HUD (MediaPipe face landmarker), consent/fairness inputs, and experiment check-ins.
- `backend/` — FastAPI WebSocket service with supportive/neutral/cold personas, heuristic tips, SQLModel persistence, export + summary endpoints.
- `docs/` — architecture/usage notes.
- `infra/` — reserved for Docker/docker-compose in a later step.

## Quick start
1) Backend
```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export DATABASE_URL="sqlite+aiosqlite:///./data.db" # or postgresql+asyncpg://app:app@localhost:5432/interview
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
2) Frontend (new shell)
```bash
cd frontend
npm install
npm run dev
```
Open `http://localhost:3000`, pick a persona, and click **Start Interview**.

## What’s included
- Style cards (supportive/neutral/cold) with style-specific follow-ups and coaching language.
- Real-time HUD powered by mic/cam: pause ratio, speaking-rate proxy, gaze via MediaPipe Tasks, fillers from text; video preview included.
- Consent + fairness inputs (accent/notes), experiment check-in controls (control vs treatment, confidence/stress sliders) with WebSocket logging.
- Voice-led loop: questions are spoken (local Coqui TTS), user answers are captured from the mic, sent to local Whisper for STT, auto-sent to the backend, and logged for latency/fairness.
- Neural TTS option: Coqui XTTS v2 endpoint (`POST /tts`) speaks questions in an adaptive tone (supportive/neutral/cold) with configurable speaker.
- Persistence for sessions/answers/check-ins/telemetry (SQLite/Postgres), aggregate deltas at `GET /metrics/summary`, and export at `GET /export/session/{id}`.
- WebSocket protocol documented in `docs/README.md`; Docker scaffolds in `infra/`; MediaPipe assets in `frontend/public/mediapipe`.

## Local Whisper STT
- Install `ffmpeg` on the host (required to decode WebM/Opus from the browser).
- The backend loads `faster-whisper` at startup. Configure via env:
  - `WHISPER_MODEL` (default `medium`, use `small`/`base` for lighter CPU)
  - `WHISPER_DEVICE` (`cuda` for GPU, default `cpu`)
  - `WHISPER_COMPUTE_TYPE` (defaults to `float16` when device is not CPU, `int8` on CPU)
- Frontend records mic audio (MediaRecorder WebM/Opus) and POSTs to `POST /stt`. Responses auto-fill the answer box and, if enabled, auto-send to the interview loop. Telemetry for STT latency is stored with the session.

## Neural TTS (Coqui XTTS)
- Install GPU deps and add `TTS` (already in `backend/requirements.txt`); ensure PyTorch with CUDA is available.
- Env:
  - `TTS_MODEL` (default `tts_models/multilingual/multi-dataset/xtts_v2`)
  - `TTS_DEVICE` (`cuda` recommended; defaults to WHISPER_DEVICE)
  - `TTS_SPEAKER` (optional speaker embedding name/path)
  - `TTS_LANGUAGE` (default `en`)
- Endpoint: `POST /tts` with JSON `{ text, style }` returns WAV audio; styles map to slight speed/pitch changes per persona. Frontend fetches this audio to speak questions.

### GPU usage
- Install CUDA drivers/toolkit compatible with your GPU.
- Set `WHISPER_DEVICE=cuda` and optionally `WHISPER_COMPUTE_TYPE=float16` (default when using CUDA).
- Ensure the `medium` model fits GPU memory (consider `small` if not). Restart the backend after changing env.

## Next steps
- Plug in ASR + richer lexical feedback.
- Add deeper fairness/latency dashboards, consent revoke/delete, and production-grade auth around exports.
- Harden Docker/compose for prod-like bring-up with env templating and metrics stack.
