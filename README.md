# Emotion-Aware AI Tutor

Minimal README to get the project running locally (frontend + backend), plus testing tips and notes about ignoring private files.

## Quick overview

This repo contains a Next.js frontend and a FastAPI backend that implements a simple emotion-aware tutor prototype. The frontend captures camera frames and sends them to the backend `/api/emotion` endpoint for emotion detection (DeepFace/OpenCV fallback).

This README focuses on local developer setup using PowerShell on Windows.

## Prerequisites

- Python 3.10+ (3.11 recommended)
- Node.js 18+ and npm
- (Optional) Git
- A webcam for live testing

## Files of note

- `frontend/` — Next.js app (React + typescript/JS) that captures camera frames and displays emotion UI.
- `backend/` — FastAPI app that handles image decoding and emotion detection.
- `.gitignore` — excludes local env files and the two project doc files `ProjectDescription.md` and `ProjectModulePlan.md` per project policy.
- `frontend/.env.example` — example frontend env showing `NEXT_PUBLIC_API_BASE`.

## Backend — Setup and run (PowerShell)

1. Create and activate a virtual environment, install dependencies.

```powershell
cd "c:\Users\mitun\OneDrive\Documents\AI Projects\emotion-aware\backend"
python -m venv env
.\env\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

2. Run the backend server (development):

```powershell
# from backend folder and with activated env
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

3. Confirm the backend is up by visiting:

- http://127.0.0.1:8000/docs (OpenAPI UI)
- http://127.0.0.1:8000/api/status

If the backend fails to start because of optional heavy deps (DeepFace, transformers, torch), either install them or accept degraded functionality (OpenCV fallback or `emotion_available` false).

## Frontend — Setup and run (PowerShell)

1. Install node modules and start dev server:

```powershell
cd "c:\Users\mitun\OneDrive\Documents\AI Projects\emotion-aware\frontend"
npm install
# start dev server, bind to 127.0.0.1
npm run dev -- --hostname 127.0.0.1
```

2. Open the app in a browser:

- http://127.0.0.1:3000

3. Configure API base (optional): copy `frontend/.env.example` to `frontend/.env.local` and edit if your backend is running on a different host/port.

## Quick end-to-end test (PowerShell + Python)

If you want to test the backend `/api/emotion` without the camera, run this small Python snippet (requires `requests` and `Pillow`): it creates a tiny image, encodes to base64, and posts to the backend.

```powershell
python - <<'PY'
import requests, base64, io
from PIL import Image
img = Image.new('RGB',(32,32),(128,128,128))
buf = io.BytesIO()
img.save(buf, format='JPEG')
b64 = 'data:image/jpeg;base64,' + base64.b64encode(buf.getvalue()).decode('ascii')
resp = requests.post('http://127.0.0.1:8000/api/emotion', json={'image': b64})
print(resp.status_code)
print(resp.text)
PY
```

You should get a JSON response like `{"emotion":"neutral","confidence":79.6,...}` (backend may return confidence in 0..100 depending on the model; the frontend normalizes values > 1 to percent).

## Debugging tips

- Browser console: watch for `Failed to fetch` — often indicates backend error (see backend logs in terminal).
- Backend logs: errors about `numpy.float32` were previously fixed by coercing outputs to native Python types before returning JSON.
- If the camera view is black, the frontend now skips sending uniformly dark frames (you'll see `No valid camera frame (dark or covered)` as the camera status).
- Use the `🖼️ Test Image` button in the UI (or the Python snippet above) to verify backend processing independent of webcam capture.

## Pushing to GitHub

Typical sequence (from repo root):

```powershell
git init
git add .
git commit -m "Initial commit: Emotion-Aware AI Tutor"
git remote add origin https://github.com/mithun0524/Emotion-Aware-AI-Tutor.git
git branch -M main
git push -u origin main
```

Remember `.gitignore` excludes `ProjectDescription.md` and `ProjectModulePlan.md` per your request. If you need those tracked, remove them from `.gitignore`.

## Recommended follow-ups

- Optionally pin versions in `backend/requirements.txt` to make installs reproducible.
- Add a lightweight GitHub Actions workflow to run lint/tests on push.
- Add a `README.dev.md` with more detailed developer notes if you plan to onboard collaborators.

If you want, I can also:

- Pin and update `backend/requirements.txt` and create a `requirements-dev.txt`.
- Create a minimal GitHub Actions workflow for linting and a smoke test.

---

If you want any of the follow-ups, tell me which and I'll add them.
# Emotion-Aware AI Tutor

An AI-powered tutor that adapts to student emotions and provides dynamic, context-aware responses.

## Project Overview

This project implements an Emotion-Aware AI Tutor with the following phases:

- **Phase 1**: Core Tutor Setup (Text-based Q&A) ✅
- **Phase 2**: Add Voice (STT + TTS)
- **Phase 3**: Emotion Awareness (CV)
- **Phase 4**: Adaptive Learning Engine
- **Phase 5**: Analytics & Progress Tracking
- **Phase 6**: Advanced Features (Optional)

## Current Status

Phase 1 is ~90% complete:
- ✅ FastAPI backend with Google GenAI (Gemini 2.5 Flash) integration
- ✅ Hugging Face fallback for local generation
- ✅ Next.js frontend with lesson interface
- ✅ Provider telemetry (shows which AI served the response)
- ✅ Secure key handling via .env

Phase 2 (Voice) is ~50% complete:
- ✅ OpenAI Whisper installed for STT
- ✅ Coqui TTS installed for voice synthesis
- ⏳ Integration with backend routes (next step)
- ⏳ Frontend voice input/output (pending)

## Setup Instructions

### Prerequisites
- Python 3.8+ (for backend)
- Node.js 16+ (for frontend)
- Rust toolchain (for tiktoken build)

### Backend Setup

1. Navigate to backend directory:
   ```bash
   cd backend
   ```

2. Create virtual environment:
   ```bash
   python -m venv env
   ```

3. Activate virtual environment:
   - Windows: `env\Scripts\activate`
   - Linux/Mac: `source env/bin/activate`

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

5. Set up Google GenAI API key:
   - Edit `backend/.env` and add your API key:
     ```
     GENAI_API_KEY=your_actual_api_key_here
     ```
   - Or set environment variable:
     ```bash
     export GENAI_API_KEY=your_key
     ```

6. Test voice components:
   ```bash
   python backend/test_stt.py  # Test Whisper STT
   python backend/test_tts.py  # Test Coqui TTS
   ```

### Frontend Setup

1. Navigate to frontend directory:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm run dev
   ```

4. Open http://localhost:3000 in your browser.

### Usage

1. Start both backend and frontend servers.
2. Navigate to the lesson page.
3. Type a question and click "Ask Tutor".
4. The response will show which provider (genai or hf) answered.

## API Endpoints

- `POST /api/tutor`: Ask a question
  - Request: `{"question": "Your question here"}`
  - Response: `{"answer": "Generated answer", "provider": "genai"}`

- `GET /api/status`: Check system status
  - Response: `{"genai_available": true, "hf_available": true, ...}`

## Architecture

- **Backend**: FastAPI with Google GenAI client and Hugging Face transformers fallback
- **Frontend**: Next.js with Tailwind CSS
- **AI Providers**:
  - Primary: Google Gemini 2.5 Flash (via google-genai SDK)
  - Fallback: Hugging Face GPT-2 (local generation)

## Security Notes

- Never commit API keys to version control
- Use .env files for local development
- Consider using secrets management for production

## Next Steps

- Phase 2: Integrate Whisper for STT and Coqui TTS for voice
- Phase 3: Add OpenCV + DeepFace for emotion detection
- Phase 4: Implement adaptive learning rules

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is open-source and available under the MIT License.
