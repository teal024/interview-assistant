# Backend (FastAPI)

Simple WebSocket-first service that powers the interview loop with style cards and heuristic coaching.

## Quick start
```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # fill in NVIDIA_API_KEY and other overrides
export DATABASE_URL="sqlite+aiosqlite:///./data.db" # or postgresql+asyncpg://app:app@localhost:5432/interview
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

WebSocket endpoint: `ws://localhost:8000/ws/interview`  
Health check: `http://localhost:8000/health`

### NVIDIA LLM credentials
Backend questions, follow-ups, and tips now use the NVIDIA chat completions API. Set `NVIDIA_API_KEY` in `.env` (see `.env.example`). You can also override `NVIDIA_LLM_MODEL`, `NVIDIA_LLM_URL`, and `NVIDIA_LLM_TIMEOUT` if needed. If the key is missing or the request fails, the service falls back to the built-in heuristics.

### Message types (WebSocket)
- `start_session` { style, group } → `session_started` + first `question`
- `user_answer` { answer, metrics } → `interviewer_message` + `tips` + next `question`
- `switch_style` { style } → `style_switched`
- `checkin` { group, confidence, stress } → `checkin_logged`
- `telemetry` { event, latencyMs, data } → stored for latency/fairness dashboards
- `ping` → `pong`

### REST additions
- `POST /checkins` — log confidence/stress (HTTP alternative to WebSocket)
- `GET /metrics/summary` — aggregated means and deltas (control vs treatment) for speaking rate, pause ratio, gaze, fillers, confidence, stress, latency.
- `GET /export/session/{id}` — export session + answers + check-ins + telemetry as JSON.
