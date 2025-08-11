import os, asyncio, time, uuid
from typing import Any, Dict, Optional
import httpx
from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

try:
    from slowapi import Limiter
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded
    from slowapi.middleware import SlowAPIMiddleware
    _slowapi_installed = True
except ImportError:
    _slowapi_installed = False

try:
    import structlog
    structlog.configure(processors=[structlog.processors.TimeStamper(fmt="iso"), structlog.processors.JSONRenderer()])
    log = structlog.get_logger()
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    log = logging.getLogger(__name__)

# --- Configuration ---
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o")
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-3-opus-20240229")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-1.5-pro-latest")
TIMEOUT_SEC = float(os.environ.get("HTTP_TIMEOUT_SEC", "90"))
MAX_RETRIES = int(os.environ.get("HTTP_MAX_RETRIES", "2"))
BACKOFF_BASE = float(os.environ.get("HTTP_BACKOFF_BASE", "1.0"))
CORS_ALLOWED = os.environ.get("CORS_ALLOW_ORIGINS", "")
INTERNAL_API_KEY = os.environ.get("INTERNAL_API_KEY")
ENABLE_RATELIMIT = os.environ.get("ENABLE_RATELIMIT", "true").lower() == "true" and _slowapi_installed
RATELIMIT_RULE = os.environ.get("RATELIMIT_RULE", "30/minute")

# --- FastAPI App & Lifespan ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.http = httpx.AsyncClient(timeout=TIMEOUT_SEC, limits=httpx.Limits(max_connections=100))
    yield
    await app.state.http.aclose()

app = FastAPI(title="지노이진호 창조명령권자 - ZINO-Genesis Engine", version="4.6 Corrected", lifespan=lifespan)

# --- Middlewares ---
app.add_middleware(CORSMiddleware, allow_origins=CORS_ALLOWED.split(",") if CORS_ALLOWED else ["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

if ENABLE_RATELIMIT:
    limiter = Limiter(key_func=get_remote_address)
    app.state.limiter = limiter
    app.add_middleware(SlowAPIMiddleware)

@app.exception_handler(RateLimitExceeded)
async def _rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(status_code=429, content={"detail": f"Too Many Requests: {exc.detail}"})

@app.middleware("http")
async def add_request_id_and_log(request: Request, call_next):
    req_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    start_time = time.monotonic()
    response = await call_next(request)
    duration_ms = (time.monotonic() - start_time) * 1000
    log.info("request_completed", request_id=req_id, method=request.method, path=request.url.path, status_code=response.status_code, duration_ms=round(duration_ms, 2))
    response.headers["X-Request-ID"] = req_id
    return response

# --- API Schemas & Health Check ---
class RouteIn(BaseModel): user_input: str
class RouteOut(BaseModel): report_md: str; meta: Dict[str, Any]
@app.get("/", tags=["Health Check"])
def health_check(): return {"status": "ok", "message": "ZINO-GE v4.6 Corrected Protocol is alive!"}

# --- Utility: Retry Logic ---
RETRY_STATUS_CODES = {429, 502, 503, 504}
async def post_with_retries(client: httpx.AsyncClient, url: str, **kwargs) -> httpx.Response:
    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = await client.post(url, **kwargs)
            if resp.status_code in RETRY_STATUS_CODES and attempt < MAX_RETRIES:
                raise httpx.HTTPStatusError(f"Retryable status: {resp.status_code}", request=resp.request, response=resp)
            resp.raise_for_status()
            return resp
        except (httpx.RequestError, httpx.HTTPStatusError) as e:
            if attempt >= MAX_RETRIES:
                log.error("http_post_failed", url=url, attempts=attempt + 1, error=str(e))
                raise
            sleep_s = BACKOFF_BASE * (2 ** attempt)
            await asyncio.sleep(sleep_s)
            log.warning("http_post_retry", url=url, attempt=attempt + 1, wait_sec=round(sleep_s, 2))
    raise RuntimeError("This should not be reached")

# --- DMAC Core Agents ---
async def call_gemini(client: httpx.AsyncClient, prompt: str) -> str:
    gemini_prompt = f"ROLE: Data Provenance Analyst... USER REQUEST: \"{prompt}\""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    payload = {"contents":[{"parts":[{"text": gemini_prompt}]}]}
    headers = {"Content-Type":"application/json"}
    r = await post_with_retries(client, url, headers=headers, json=payload)
    return r.json()["candidates"][0]["content"]["parts"][0]["text"]

async def call_claude(client: httpx.AsyncClient, prompt: str) -> str:
    claude_prompt = f"ROLE: Strategic Foresight Simulator... USER REQUEST: \"{prompt}\""
    url = "https://api.anthropic.com/v1/messages"
    headers = {"x-api-key": ANTHROPIC_API_KEY, "anthropic-version": "2023-06-01", "content-type": "application/json"}
    payload = {"model": ANTHROPIC_MODEL, "max_tokens": 4096, "messages": [{"role": "user", "content": claude_prompt}]}
    r = await post_with_retries(client, url, headers=headers, json=payload)
    return "".join([b.get("text", "") for b in r.json().get("content", [])])

async def call_gpt_creative(client: httpx.AsyncClient, prompt: str) -> str:
    gpt_prompt = f"ROLE: Creative Challenger... USER REQUEST: \"{prompt}\""
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": OPENAI_MODEL, "messages": [{"role": "user", "content": gpt_prompt}], "temperature": 0.7}
    r = await post_with_retries(client, url, headers=headers, json=payload)
    return r.json()["choices"][0]["message"]["content"]

async def call_gpt_orchestrator(client: httpx.AsyncClient, original_prompt: str, reports: list[str]) -> str:
    system_prompt = "You are 'The First Cause: Quantum Oracle'..."
    user_prompt = f"Original User Directive: \"{original_prompt}\"\n---\n[Report 1]...{reports[0]}\n---\n[Report 2]...{reports[1]}\n---\n[Report 3]...{reports[2]}\n---\nSynthesize."
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": OPENAI_MODEL, "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}], "temperature": 0.1}
    r = await post_with_retries(client, url, headers=headers, json=payload)
    return r.json()["choices"][0]["message"]["content"]

# --- Main Route ---
@app.post("/route", response_model=RouteOut, tags=["ZINO-GE Core v4.6 Corrected"])
async def route(
    payload: RouteIn,
    request: Request,
    x_internal_api_key: Optional[str] = Header(default=None, alias="X-Internal-API-Key"),
):
    if INTERNAL_API_KEY and x_internal_api_key != INTERNAL_API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized: Invalid internal API key")
    
    if ENABLE_RATELIMIT and _slowapi_installed:
        limiter = request.app.state.limiter
        await limiter.hit(request) # This is the corrected line

    if not all([OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY]):
        raise HTTPException(status_code=500, detail="Server configuration error: API keys are missing.")

    client: httpx.AsyncClient = request.app.state.http
    
    # ... (rest of the function is the same) ...
