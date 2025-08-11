import os, asyncio, time, uuid
from typing import Any, Dict, List, Optional
import httpx
from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# --- Optional Dependencies ---
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

# ================== Configuration ==================
# Model Names
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5")
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-3-5-sonnet-20240620")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-1.5-pro-latest")

# Network & Retry Settings
TIMEOUT_SEC = float(os.environ.get("HTTP_TIMEOUT_SEC", 90.0))
MAX_RETRIES = int(os.environ.get("HTTP_MAX_RETRIES", 2))
BACKOFF_BASE = float(os.environ.get("HTTP_BACKOFF_BASE", 1.0))

# Security & Rate Limiting
CORS_ALLOWED = os.environ.get("CORS_ALLOW_ORIGINS", "")
INTERNAL_API_KEY_STORE = os.environ.get("INTERNAL_API_KEY")
ENABLE_RATELIMIT = os.environ.get("ENABLE_RATELIMIT", "true").lower() == "true" and _slowapi_installed
RATELIMIT_RULE = os.environ.get("RATELIMIT_RULE", "30/minute")

# ================== FastAPI App & Lifespan ==================
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.http = httpx.AsyncClient(timeout=TIMEOUT_SEC, limits=httpx.Limits(max_connections=100))
    yield
    await app.state.http.aclose()

app = FastAPI(title="지노이진호 창조명령권자 - ZINO-Genesis Engine", version="10.0 Absolute", lifespan=lifespan)

# ================== Middlewares ==================
app.add_middleware(CORSMiddleware, allow_origins=CORS_ALLOWED.split(",") if CORS_ALLOWED else ["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

if _slowapi_installed and ENABLE_RATELIMIT:
    limiter = Limiter(key_func=get_remote_address)
    app.state.limiter = limiter
    app.add_middleware(SlowAPIMiddleware)

    @app.exception_handler(RateLimitExceeded)
    async def _rate_limit_handler(request: Request, exc: RateLimitExceeded):
        log.warning("rate_limit_exceeded", remote_addr=get_remote_address(request), detail=exc.detail)
        return JSONResponse(status_code=429, content={"detail": f"Too Many Requests: {exc.detail}"})

# ================== API Schemas ==================
class RouteIn(BaseModel): user_input: str
class RouteOut(BaseModel): report_md: str; meta: Dict[str, Any]

# ================== Diagnostic Endpoints ==================
@app.get("/", tags=["Health Check"])
def health_check(): return {"status": "ok", "message": "ZINO-GE v10.0 Absolute Protocol is alive!"}

@app.get("/version", tags=["Diagnostics"])
def version(): return {"app":"ZINO-GE","version":"10.0 Absolute","models":{"openai":OPENAI_MODEL, "anthropic":ANTHROPIC_MODEL, "gemini":GEMINI_MODEL}}

# ================== Utility Functions ==================
def safe_get(d: Dict, path: list, default: Any = "") -> Any:
    cur = d
    try:
        for k in path:
            if isinstance(cur, list) and isinstance(k, int): cur = cur[k]
            elif isinstance(cur, dict): cur = cur.get(k, {})
            else: return default
        return cur if isinstance(cur, str) else (str(cur) if cur not in (None, {}) else default)
    except Exception:
        return default

RETRY_STATUS_CODES = {429, 502, 503, 504}
async def post_with_retries(client: httpx.AsyncClient, agent_name: str, url: str, **kwargs) -> httpx.Response:
    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = await client.post(url, **kwargs)
            if resp.status_code in RETRY_STATUS_CODES and attempt < MAX_RETRIES:
                raise httpx.HTTPStatusError(f"Retryable status: {resp.status_code}", request=resp.request, response=resp)
            resp.raise_for_status()
            return resp
        except (httpx.RequestError, httpx.HTTPStatusError) as e:
            log.warning("http_post_retry", agent=agent_name, url=url, attempt=attempt + 1, error=str(e))
            if attempt >= MAX_RETRIES:
                log.error("http_post_failed", agent=agent_name, url=url, attempts=attempt + 1, error=str(e))
                raise
            sleep_s = BACKOFF_BASE * (2 ** attempt)
            await asyncio.sleep(sleep_s)
    raise RuntimeError("Retry logic should not reach this point")

# ================== DMAC Core Agents ==================
O_HEADERS = lambda: {"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}", "Content-Type": "application/json"}

async def call_gemini(client: httpx.AsyncClient, prompt: str) -> str:
    gemini_prompt = f"ROLE: Data Provenance Analyst. AXIOM: Data-First. USER REQUEST: \"{prompt}\""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={os.environ.get('GEMINI_API_KEY')}"
    payload = {"contents":[{"parts":[{"text": gemini_prompt}]}]}
    headers = {"Content-Type":"application/json"}
    r = await post_with_retries(client, "Gemini", url, headers=headers, json=payload)
    return safe_get(r.json(), ["candidates", 0, "content", "parts", 0, "text"], default="[GEMINI_EMPTY_RESPONSE]")

async def call_claude(client: httpx.AsyncClient, prompt: str) -> str:
    claude_prompt = f"ROLE: Strategic Foresight Simulator. FRAMEWORK: QVF v2.0. USER REQUEST: \"{prompt}\""
    url = "https://api.anthropic.com/v1/messages"
    headers = {"x-api-key": os.environ.get("ANTHROPIC_API_KEY"), "anthropic-version":"2023-06-01", "content-type":"application/json"}
    payload = {"model": ANTHROPIC_MODEL, "max_tokens": 2048, "messages":[{"role":"user","content": claude_prompt}]}
    r = await post_with_retries(client, "Claude", url, headers=headers, json=payload)
    parts = r.json().get("content", [])
    return "".join([b.get("text","") for b in parts]) or "[CLAUDE_EMPTY_RESPONSE]"

async def call_gpt_creative(client: httpx.AsyncClient, prompt: str) -> str:
    gpt_prompt = f"ROLE: Creative Challenger. Provide unconventional strategies. USER REQUEST: \"{prompt}\""
    body = {"model": OPENAI_MODEL, "input": [{"role":"user","content":gpt_prompt}], "temperature": 0.7}
    r = await post_with_retries(client, "GPT-Creative", "https://api.openai.com/v1/responses", headers=O_HEADERS(), json=body)
    return safe_get(r.json(), ["output_text"], default="[OPENAI_CREATIVE_EMPTY]")

async def call_gpt_orchestrator(client: httpx.AsyncClient, original_prompt: str, reports: List[str]) -> str:
    system_prompt = "You are 'The First Cause: Quantum Oracle', the final executor of the GCI. Synthesize the following three independent expert reports into a single, final, actionable Genesis Command for the '창조명령권자 지노이진호'. Your synthesis must be cross-validated against the 3 Axioms and serve the Top-level Directive: '레독스톤(이오나이트) 사업의 성공'. IMPORTANT: The final output MUST be written entirely in Korean."
    user_prompt = f"Original User Directive: \"{original_prompt}\"\n---\n[Report 1: Data Provenance]\n{reports[0]}\n---\n[Report 2: Strategic Simulation]\n{reports[1]}\n---\n[Report 3: Creative Alternatives]\n{reports[2]}\n---\nSynthesize the final Genesis Command."
    body = {"model": OPENAI_MODEL, "input": [{"role":"system","content":system_prompt}, {"role":"user","content":user_prompt}], "temperature": 0.1}
    r = await post_with_retries(client, "GPT-Orchestrator", "https://api.openai.com/v1/responses", headers=O_HEADERS(), json=body)
    return safe_get(r.json(), ["output_text"], default="[OPENAI_ORCHESTRATOR_EMPTY]")

# ================== Main Route ==================
@app.post("/route", response_model=RouteOut, tags=["ZINO-GE Core v10.0 Absolute"])
async def route(
    payload: RouteIn,
    request: Request,
    x_internal_api_key: Optional[str] = Header(default=None, alias="X-Internal-API-Key"),
):
    if INTERNAL_API_KEY_STORE and x_internal_api_key != INTERNAL_API_KEY_STORE:
        raise HTTPException(status_code=401, detail="Unauthorized: Invalid internal API key")
    
    # ✅ PATCH: 키 존재 여부를 요청 시점에 다시 확인 (롤오버 대응)
    if not all([os.getenv("OPENAI_API_KEY"), os.getenv("ANTHROPIC_API_KEY"), os.getenv("GEMINI_API_KEY")]):
        return RouteOut(
            report_md="## 시스템 오류\n필수 API 키 일부가 설정되지 않았습니다. Render 대시보드의 Environment 탭을 확인하십시오.",
            meta={"error":"SERVER_CONFIG_MISSING_KEYS"}
        )

    if _slowapi_installed and ENABLE_RATELIMIT:
        await request.app.state.limiter.hit(RATELIMIT_RULE, request)

    client: httpx.AsyncClient = request.app.state.http

    tasks = [
        call_gemini(client, payload.user_input),
        call_claude(client, payload.user_input),
        call_gpt_creative(client, payload.user_input),
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    def unwrap(res: Any, agent_name: str) -> str:
        if isinstance(res, Exception):
            log.error("agent_call_failed", agent=agent_name, error=str(res))
            return f"[{agent_name} 에이전트 오류: {type(res).__name__}]"
        return res

    gemini_res = unwrap(results[0], "Gemini")
    claude_res = unwrap(results[1], "Claude")
    gpt_res = unwrap(results[2], "GPT-Creative")

    try:
        final_report = await call_gpt_orchestrator(client, payload.user_input, [gemini_res, claude_res, gpt_res])
    except Exception as e:
        log.exception("orchestration_failed", error=str(e))
        final_report = (
            f"## 최종 종합 실패\n\n- **오류 원인:** {type(e).__name__}\n\n"
            "### 개별 에이전트 보고서 (요약):\n"
            f"**Gemini:**\n```\n{gemini_res[:500]}...\n```\n"
            f"**Claude:**\n```\n{claude_res[:500]}...\n```\n"
            f"**GPT-Creative:**\n```\n{gpt_res[:500]}...\n```"
        )

    meta_data = {
        "gemini_report": gemini_res,
        "claude_report": claude_res,
        "gpt_creative_report": gpt_res,
    }
    return RouteOut(report_md=final_report, meta=meta_data)
