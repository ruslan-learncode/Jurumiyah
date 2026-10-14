"""Small API backend for the Ajrumiyyah Pro chatbot."""

import os
import logging
import json
import re
import time
import threading
import uuid
from collections import OrderedDict, defaultdict, deque
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from google import genai
from google.genai import types
from pydantic import BaseModel, Field


app = FastAPI(title="Ajrumiyyah Pro Chat API")
logger = logging.getLogger(__name__)
gemini_api_key = os.getenv("GEMINI_API_KEY", "").strip().strip('"').strip("'")
if not gemini_api_key:
    raise RuntimeError("GEMINI_API_KEY belum diatur pada environment backend.")
client = genai.Client(
    api_key=gemini_api_key,
    http_options=types.HttpOptions(
        api_version="v1",
        timeout=45_000,
        retry_options=types.HttpRetryOptions(
            attempts=2,
            initial_delay=0.5,
            max_delay=1.0,
            http_status_codes=[408, 429, 500, 502, 503, 504],
        ),
    ),
)
model = os.getenv("GEMINI_MODEL", "gemini-3.8-flash").strip()

# The HTML is served from this app, so same-origin requests need no CORS.
# Add exact origins to CHAT_ALLOWED_ORIGINS only if hosting the frontend elsewhere.
allowed_origins = [
    origin.strip()
    for origin in os.getenv("CHAT_ALLOWED_ORIGINS", "").split(",")
    if origin.strip()
]
if allowed_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_methods=["POST"],
        allow_headers=["Content-Type"],
        expose_headers=["X-Chat-Request-ID"],
    )

RATE_LIMIT = 10
RATE_WINDOW_SECONDS = 60
requests_by_ip: dict[str, deque[float]] = defaultdict(deque)
CACHE_TTL_SECONDS = 15 * 60
CACHE_MAX_ENTRIES = 256
answer_cache: OrderedDict[str, tuple[float, str]] = OrderedDict()
answer_cache_lock = threading.Lock()


def cache_key(message: str) -> str:
    """Normalize whitespace and case so simple variants share a cached answer."""
    return re.sub(r"\s+", " ", message).strip().casefold()


def get_cached_answer(key: str) -> str | None:
    now = time.monotonic()
    with answer_cache_lock:
        cached = answer_cache.get(key)
        if cached is None:
            return None
        expires_at, answer = cached
        if expires_at <= now:
            del answer_cache[key]
            return None
        answer_cache.move_to_end(key)
        return answer


def store_cached_answer(key: str, answer: str) -> None:
    with answer_cache_lock:
        answer_cache[key] = (time.monotonic() + CACHE_TTL_SECONDS, answer)
        answer_cache.move_to_end(key)
        while len(answer_cache) > CACHE_MAX_ENTRIES:
            answer_cache.popitem(last=False)


def ndjson_event(**event: str | bool) -> str:
    return json.dumps(event, ensure_ascii=False) + "\n"


def model_error_message(exc: Exception) -> str:
    detail = str(exc)
    if gemini_api_key:
        detail = detail.replace(gemini_api_key, "[API key disembunyikan]")
    return detail.strip()[:500] or "Gemini gagal memproses permintaan."


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(Path(__file__).with_name("ajrumiyyah-pro.html"))


@app.post("/api/chat")
def chat(payload: ChatRequest, request: Request):
    message = payload.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Pesan masih kosong.")

    key = cache_key(message)
    request_id = uuid.uuid4().hex[:12]
    cached_answer = get_cached_answer(key)
    if cached_answer is not None:
        def cached_stream():
            yield ndjson_event(cache=True)
            yield ndjson_event(delta=cached_answer)
            yield ndjson_event(done=True)

        logger.info("chat request_id=%s cache_hit=true", request_id)
        return StreamingResponse(
            cached_stream(),
            media_type="application/x-ndjson",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "X-Chat-Request-ID": request_id,
            },
        )

    now = time.monotonic()
    client_ip = request.client.host if request.client else "unknown"
    recent = requests_by_ip[client_ip]
    while recent and now - recent[0] >= RATE_WINDOW_SECONDS:
        recent.popleft()
    if len(recent) >= RATE_LIMIT:
        raise HTTPException(
            status_code=429,
            detail="Terlalu banyak pesan. Silakan tunggu sebentar lalu coba lagi.",
        )
    recent.append(now)

    def generate_answer():
        started_at = time.perf_counter()
        answer_parts: list[str] = []
        stream_open_ms = None
        first_text_ms = None
        try:
            logger.info("chat request_id=%s gemini_request_started=true model=%s", request_id, model)
            chunks = client.models.generate_content_stream(
                model=model,
                contents=message,
                config=types.GenerateContentConfig(
                    system_instruction=(
                        "Kamu adalah asisten belajar yang ramah dan menjawab dalam bahasa "
                        "yang digunakan pengguna, terutama bahasa Indonesia. Jawab sapaan "
                        "dan pertanyaan umum secara langsung. Kamu juga membantu belajar "
                        "Islam, nahwu, dan sharaf. Untuk pertanyaan agama yang memerlukan "
                        "fatwa atau memiliki perbedaan pendapat, jelaskan keterbatasanmu "
                        "dan sarankan pengguna bertanya kepada ahli tepercaya. Jika tidak "
                        "yakin, katakan dengan jujur. Gunakan teks biasa, ringkas, dan "
                        "jangan keluarkan HTML."
                    ),
                    max_output_tokens=350,
                    temperature=0.3,
                    thinking_config=types.ThinkingConfig(thinking_level="low"),
                ),
            )
            stream_open_ms = (time.perf_counter() - started_at) * 1000
            yield ndjson_event(cache=False)
            for chunk in chunks:
                text = chunk.text or ""
                if text:
                    if first_text_ms is None:
                        first_text_ms = (time.perf_counter() - started_at) * 1000
                    answer_parts.append(text)
                    yield ndjson_event(delta=text)

            answer = "".join(answer_parts).strip()
            if not answer:
                logger.info(
                    "chat request_id=%s cache_hit=false gemini_stream_open_ms=%.1f "
                    "first_text_ms=none total_ms=%.1f empty=true",
                    request_id,
                    stream_open_ms,
                    (time.perf_counter() - started_at) * 1000,
                )
                yield ndjson_event(error="Model tidak mengirim jawaban.")
                return
            store_cached_answer(key, answer)
            logger.info(
                "chat request_id=%s cache_hit=false gemini_stream_open_ms=%.1f "
                "first_text_ms=%.1f total_ms=%.1f",
                request_id,
                stream_open_ms,
                first_text_ms,
                (time.perf_counter() - started_at) * 1000,
            )
            yield ndjson_event(done=True)
        except Exception as exc:
            logger.exception("Chat model request failed")
            logger.info(
                "chat request_id=%s cache_hit=false gemini_stream_open_ms=%s "
                "first_text_ms=%s total_ms=%.1f error=true",
                request_id,
                f"{stream_open_ms:.1f}" if stream_open_ms is not None else "before-open",
                f"{first_text_ms:.1f}" if first_text_ms is not None else "none",
                (time.perf_counter() - started_at) * 1000,
            )
            yield ndjson_event(error=f"Gemini API gagal: {model_error_message(exc)}")

    return StreamingResponse(
        generate_answer(),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "X-Chat-Request-ID": request_id,
        },
    )
