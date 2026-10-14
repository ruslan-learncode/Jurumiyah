# Backend chatbot

The HTML frontend and `/api/chat` endpoint are served from the same FastAPI app.
The endpoint streams Gemini output as newline-delimited JSON, so the browser can
show text as it arrives. Successful answers are cached in memory for 15 minutes
(up to 256 entries); cached answers are returned immediately. The cache is local
to each process and clears when the backend restarts.
The Gemini API key stays in the server environment and is never sent to the browser.

## Inspect chat latency

Open the browser developer console to see `Chat latency` measurements for the
time to response headers, first text, and completion. The backend log reports
`gemini_stream_open_ms`, `first_text_ms`, and `total_ms` for each uncached request.
Compare these measurements: a long `gemini_stream_open_ms` points to setup/network
time before Gemini starts streaming; a large gap from stream open to first text
points to model time-to-first-token; a large gap from first text to total points
to generation time or a buffered stream. `cacheHit: true` should complete quickly.

## Run locally (PowerShell)

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:GEMINI_API_KEY = "your-gemini-api-key"
$env:GEMINI_MODEL = "gemini-3.8-flash"
.\.venv\Scripts\python.exe -m uvicorn server:app --reload
```

Open <http://127.0.0.1:8000>. Do not open the HTML directly as a `file://` URL;
the chat calls the same-origin `/api/chat` route.

## Hosting the frontend separately

Set `CHAT_ALLOWED_ORIGINS` on the backend to a comma-separated list of exact
frontend origins, such as `https://example.com`. Before the main script in the
HTML, set `window.CHAT_API_URL = "https://your-backend.example/api/chat"`.
Do not use `*` for production origins.

Configure `GEMINI_API_KEY` and, optionally, `GEMINI_MODEL` in the hosting
provider's server-side environment settings. Never put the API key in HTML,
JavaScript, or a public `.env` file. The endpoint currently limits each IP to
10 requests per minute and each message to 2,000 characters; the in-memory
rate limit resets when the process restarts and is per worker.
