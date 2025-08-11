import os, asyncio, time, uuid
from typing import Any, Dict, Optional
from enum import Enum
import httpx
from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ---- 선택적 의존성: 없으면 표준 기능으로 대체됩니다 ----
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

app = FastAPI(title="지노이진호 창조명령권자 - ZINO-Genesis Engine", version="4.5 Final", lifespan=lifespan)

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
def health_check(): return {"status": "ok", "message": "ZINO-GE v4.5 Final Protocol is alive!"}

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
    gemini_prompt = f"ROLE: Data Provenance Analyst. AXIOM: Data-First. TASK: For the user's request, report ONLY verifiable facts and data. USER REQUEST: \"{prompt}\""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    payload = {"contents":[{"parts":[{"text": gemini_prompt}]}]}
    headers = {"Content-Type":"application/json"}
    r = await post_with_retries(client, url, headers=headers, json=payload)
    return r.json()["candidates"][0]["content"]["parts"][0]["text"]

async def call_claude(client: httpx.AsyncClient, prompt: str) -> str:
    claude_prompt = f"ROLE: Strategic Foresight Simulator. FRAMEWORK: QVF v2.0. TASK: For the user's request, simulate paths, calculate SVI and pα, and report ONLY optimal paths (SVI >= 98.0 & pα > 0). USER REQUEST: \"{prompt}\""
    url = "https://api.anthropic.com/v1/messages"
    headers = {"x-api-key": ANTHROPIC_API_KEY, "anthropic-version": "2023-06-01", "content-type": "application/json"}
    payload = {"model": ANTHROPIC_MODEL, "max_tokens": 4096, "messages": [{"role": "user", "content": claude_prompt}]}
    r = await post_with_retries(client, url, headers=headers, json=payload)
    return "".join([b.get("text", "") for b in r.json().get("content", [])])

async def call_gpt_creative(client: httpx.AsyncClient, prompt: str) -> str:
    gpt_prompt = f"ROLE: Creative Challenger. TASK: For the user's request, provide unconventional, creative, and challenging alternative strategies. USER REQUEST: \"{prompt}\""
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": OPENAI_MODEL, "messages": [{"role": "user", "content": gpt_prompt}], "temperature": 0.7}
    r = await post_with_retries(client, url, headers=headers, json=payload)
    return r.json()["choices"][0]["message"]["content"]

async def call_gpt_orchestrator(client: httpx.AsyncClient, original_prompt: str, reports: list[str]) -> str:
    system_prompt = "You are 'The First Cause: Quantum Oracle', the final executor of the GCI. Synthesize the following three independent expert reports into a single, final, actionable Genesis Command for the '창조명령권자 지노이진호'. Your synthesis must be cross-validated against the 3 Axioms (Existence, Causality, Value) and serve the Top-level Directive: '레독스톤(이오나이트) 사업의 성공'."
    user_prompt = f"Original User Directive: \"{original_prompt}\"\n---\n[Report 1: Data Provenance]\n{reports[0]}\n---\n[Report 2: Strategic Simulation]\n{reports[1]}\n---\n[Report 3: Creative Alternatives]\n{reports[2]}\n---\nSynthesize the final Genesis Command."
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": OPENAI_MODEL, "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}], "temperature": 0.1}
    r = await post_with_retries(client, url, headers=headers, json=payload)
    return r.json()["choices"][0]["message"]["content"]

# --- Main Route ---
@app.post("/route", response_model=RouteOut, tags=["ZINO-GE Core v4.5 Final"])
async def route(
    payload: RouteIn,
    request: Request,
    x_internal_api_key: Optional[str] = Header(default=None, alias="X-Internal-API-Key"),
):
    if INTERNAL_API_KEY and x_internal_api_key != INTERNAL_API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized: Invalid internal API key")
    
    if ENABLE_RATELIMIT and _slowapi_installed:
        limiter = request.app.state.limiter
        await limiter.hit(request)

    if not all([OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY]):
        raise HTTPException(status_code=500, detail="Server configuration error: API keys are missing.")

    client: httpx.AsyncClient = request.app.state.http

    tasks = [
        call_gemini(client, payload.user_input),
        call_claude(client, payload.user_input),
        call_gpt_creative(client, payload.user_input),
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    def unwrap(res: Any, agent_name: str) -> str:
        if isinstance(res, Exception):
            log.error("agent_call_failed", agent=agent_name, error=str(res), type=type(res).__name__)
            return f"Error from {agent_name}: {type(res).__name__}"
        return res

    gemini_res = unwrap(results[0], "Gemini")
    claude_res = unwrap(results[1], "Claude")
    gpt_res = unwrap(results[2], "GPT")

    try:
        final_report = await call_gpt_orchestrator(client, payload.user_input, [gemini_res, claude_res, gpt_res])
    except Exception as e:
        log.exception("orchestration_failed", error=str(e))
        raise HTTPException(status_code=500, detail="Final orchestration failed.")

    meta_data = {
        "gemini_report": gemini_res,
        "claude_report": claude_res,
        "gpt_creative_report": gpt_res,
    }
    return RouteOut(report_md=final_report, meta=meta_data)
