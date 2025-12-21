# AI Interview Trainer — Notes

## Architecture (current scaffold)
- **Frontend:** Next.js (App Router, TypeScript). WebSocket client for interview loop, style switcher (supportive/neutral/cold), HUD with live WebRTC/WebAudio telemetry (pause ratio, speaking-rate proxy, gaze via MediaPipe Tasks), consent/fairness inputs, and experiment check-in controls.
- **Backend:** FastAPI WebSocket endpoint (`/ws/interview`) with style-aware prompts and heuristic coaching; health check at `/health`; async persistence via SQLModel to SQLite/Postgres; metrics summary endpoint at `/metrics/summary`; export endpoint `/export/session/{id}`.
- **Env:** `frontend/.env.local.example` points `NEXT_PUBLIC_WS_URL` to `ws://localhost:8000/ws/interview`.

## Local run
```bash
# Backend
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export DATABASE_URL="sqlite+aiosqlite:///./data.db" # or postgresql+asyncpg://app:app@localhost:5432/interview
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Frontend (new shell)
cd frontend
npm install
npm run dev
```
Visit `http://localhost:3000` and click **Start Interview**.

## Message protocol (beta)
- `session_ready` (server → client): { session_id, style }
- `start_session` (client → server): { type: "start_session", style }
- `question` (server → client): { type: "question", turn, question, style, preface? } (optional short acknowledgement spoken before the question)
- `user_answer` (client → server): { type: "user_answer", answer, metrics }
- `interviewer_message` (server → client): { type: "interviewer_message", turn, message, style }
- `tips` (server → client): { type: "tips", turn, items: [{summary, detail}] }
- `switch_style` (client → server): { type: "switch_style", style }
- `style_switched` (server → client): { type: "style_switched", style }
- `telemetry` (client → server): { type: "telemetry", event, latencyMs, data } (e.g., round-trip latency per turn)
- `checkin` (client → server): { type: "checkin", group, confidence, stress } → `checkin_logged`
- `ping`/`pong` for keepalive.

## Experiment flow scaffold
- On the right column, use **Check-ins** to tag group (control vs treatment) and log confidence/stress (0–100) during pre-test, training, and post-tests.
- Transcript + tips cards are style-aware; per-turn tips are capped to two actionable items.

## Next implementation steps
- Plug ASR/LLM providers and enrich metrics with lexical content; keep MediaPipe Tasks for gaze/head-pose.
- Telemetry: log latency per module, ASR WER per accent (prosody-only fallback), and feedback acceptance rate.
- Harden the experiment pipeline with scripts to export effect sizes and confidence intervals, plus consent revoke/delete flows and auth around exports.
