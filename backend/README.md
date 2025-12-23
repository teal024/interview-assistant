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
export NUMBA_CACHE_DIR="/tmp/numba" # if TTS/librosa import hits a numba cache error
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Conda (ml311)
This repo includes a `environment.yml` (at the repo root) that can be used instead of `venv`.
```bash
conda env create -f environment.yml
conda activate ml311
cd backend
cp .env.example .env
export NUMBA_CACHE_DIR="/tmp/numba" # if TTS/librosa import hits a numba cache error
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

WebSocket endpoint: `ws://localhost:8000/ws/interview`  
Health check: `http://localhost:8000/health`

## Docker
Build (from repo root):
```bash
docker build -t interview-assistant-backend:local -f backend/Dockerfile backend
```

Run with SQLite persisted in a Docker volume:
```bash
docker run --rm -p 8000:8000 \
  --env-file backend/.env \
  -e DATABASE_URL=sqlite+aiosqlite:////data/data.db \
  -v interview_backend_data:/data \
  interview-assistant-backend:local
```

### GPU (CUDA)
Build a GPU-enabled image:
```bash
docker build -t interview-assistant-backend:gpu -f backend/Dockerfile.gpu backend
```

Run on a GPU server (requires NVIDIA Container Toolkit):
```bash
docker run --rm --gpus all -p 8000:8000 \
  --env-file backend/.env \
  -e DATABASE_URL=sqlite+aiosqlite:////data/data.db \
  -v interview_backend_data:/data \
  interview-assistant-backend:gpu
```

Verify it’s using the GPU by checking logs for `on cuda` (Whisper/TTS), or:
```bash
docker exec -it <container_id> python -c "import torch; print(torch.cuda.is_available())"
```

### Push to Docker Hub
```bash
docker login
docker tag interview-assistant-backend:local <dockerhub_user>/<repo>:latest
docker push <dockerhub_user>/<repo>:latest

# GPU tag
docker tag interview-assistant-backend:gpu <dockerhub_user>/<repo>:gpu
docker push <dockerhub_user>/<repo>:gpu
```

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
