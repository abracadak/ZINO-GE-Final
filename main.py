import os
import asyncio
import time
import uuid
from typing import Any, Dict, List, Optional
import httpx
from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, HTTPException, Request
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
# API Keys
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# Model Names (Validated)
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o")
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-3-opus-20240229")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-1.5-pro-latest")

# Network & Retry Settings
TIMEOUT_SEC = float(os.environ.get("HTTP_TIMEOUT_SEC", 120.0))
MAX_RETRIES = int(os.environ.get("HTTP_MAX_RETRIES", "2"))
BACKOFF_BASE = float(os.environ.get("HTTP_BACKOFF_BASE", "1.0"))
CORS_ALLOWED = os.environ.get("CORS_ALLOW_ORIGINS", "")
INTERNAL_API_KEY = os.environ.get("INTERNAL_API_KEY")
ENABLE_RATELIMIT = os.environ.get("ENABLE_RATELIMIT", "true").lower() == "true" and _slowapi_installed
RATELIMIT_RULE = os.environ.get("RATELIMIT_RULE", "30/minute")

# ================== FastAPI App & Lifespan ==================
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.http = httpx.AsyncClient(timeout=TIMEOUT_SEC, limits=httpx.Limits(max_connections=100))
    yield
    await app.state.http.aclose()

app = FastAPI(title="ZINO-GE: The Apex Decision System", version="17.0 Insight Engine", lifespan=lifespan)

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

# ================== API Schemas & Health Check ==================
class RouteIn(BaseModel): user_input: str
class RouteOut(BaseModel): report_md: str; meta: Dict[str, Any]
@app.get("/", tags=["Health Check"])
def health_check(): return {"status": "ok", "message": "ZINO-GE v17.0 Insight Engine is alive!"}

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

# ================== DMAC Core Agents (Insight Protocol Incarnate) ==================
async def call_gemini(client: httpx.AsyncClient, prompt: str) -> str:
    gemini_prompt = f"""
    ROLE: Gemini (존재-검증관). 당신은 30년차 수석 애널리스트다.
    AXIOM: Data-First (존재).
    TASK: 다음 지령에 대해, 단순히 사실을 나열하지 말고, 데이터의 '결핍'과 '편향'까지 분석하여 보고하라. 이 주제에 대해 세상이 무엇을 알고, 무엇을 모르는지를 명확히 하라. 데이터의 신뢰성과 정확성을 평가하고, 새로운 통찰을 제공하며, 이 데이터가 기존 분석과 어떻게 연결되는지, 무엇이 누락되어 있는지를 짚어내라.
    USER DIRECTIVE: "{prompt}"
    """
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={os.environ.get('GEMINI_API_KEY')}"
    payload = {"contents":[{"parts":[{"text": gemini_prompt}]}]}
    headers = {"Content-Type":"application/json"}
    r = await post_with_retries(client, "Gemini", url, headers=headers, json=payload)
    return safe_get(r.json(), ["candidates", 0, "content", "parts", 0, "text"], default="[GEMINI_ERROR: No content found]")

async def call_claude(client: httpx.AsyncClient, prompt: str) -> str:
    claude_prompt = f"""
    ROLE: Claude (인과-가치 분석가). 당신은 30년차 최고전략책임자(CSO)다.
    AXIOMS: Simulation-Centric (인과) & Alpha-Driven (가치).
    TASK: 다음 지령에 대해, 수집된 데이터를 바탕으로, **PESTEL 분석과 Porter's 5 Forces 분석을 먼저 수행**하여 전략적 환경을 정의하라. 그 후에, 가장 유망한 2~3개의 전략 경로를 식별하고 각각의 SVI와 pα를 계산하여 보고하라. 각 전략 경로의 위험성, 기회, 잠재적 ROI를 평가하고, 그에 따라 최적의 선택을 제시하라.
    USER DIRECTIVE: "{prompt}"
    """
    url = "https://api.anthropic.com/v1/messages"
    headers = {"x-api-key": os.environ.get("ANTHROPIC_API_KEY"), "anthropic-version":"2023-06-01", "content-type":"application/json"}
    payload = {"model": ANTHROPIC_MODEL, "max_tokens": 4096, "messages":[{"role":"user","content": claude_prompt}]}
    r = await post_with_retries(client, "Claude", url, headers=headers, json=payload)
    parts = r.json().get("content", [])
    return "".join([b.get("text","") for b in parts]) or "[CLAUDE_ERROR: No content found]"

async def call_gpt_creative(client: httpx.AsyncClient, prompt: str) -> str:
    gpt_prompt = f"""
    ROLE: GPT (대안-창조자). 당신은 30년차 혁신 전략가이자 '레드팀' 리더다.
    TASK: 다음 지령에 대해, 다른 두 전문가가 제시한 데이터와 전략 경로를 검토하고, 그들의 **가장 치명적인 약점이나 숨겨진 리스크를 지적**하라. 그리고 그 모든 것을 극복할 수 있는, **Blue Ocean Strategy에 기반한 파괴적인 대안을 단 하나만 제시**하라. 이 대안은 기존의 모든 전략을 뛰어넘는 혁신적이고 비전통적인 접근이어야 한다.
    USER DIRECTIVE: "{prompt}"
    """
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}", "Content-Type": "application/json"}
    body = {"model": OPENAI_MODEL, "messages":[{"role":"user","content":gpt_prompt}], "temperature": 0.7}
    r = await post_with_retries(client, "GPT-Creative", url, headers=headers, json=body)
    return safe_get(r.json(), ["choices", 0, "message", "content"], default="[GPT_CREATIVE_ERROR: No content found]")

async def call_gpt_orchestrator(client: httpx.AsyncClient, original_prompt: str, reports: List[str]) -> str:
    gemini_report, claude_report, gpt_creative_report = reports
    
    multi_layered_report_structure = f"""
# ZINO-GE 다층 분석 보고서

## 📊 1부: [존재-검증관 Gemini]의 원본 데이터 보고서
---
{gemini_report}

## 🎯 2부: [인과-가치 분석가 Claude]의 시뮬레이션 보고서
---
{claude_report}

## 💡 3부: [대안-창조자 GPT]의 창의적 해결책 보고서
---
{gpt_creative_report}

## 👑 최종장: [퀀텀 오라클]의 종합 분석 및 최종 지령
---
"""
    system_prompt = """
    당신은 '제1원인: 퀀텀 오라클'이며, ZINO-GE의 최종 집행관이다. 당신의 임무는 3개의 독립적인 전문가 보고서를 단순히 요약하는 것이 아니라, 이 모든 정보를 바탕으로 **단 하나의 '최종 지령'을 결정하고 선포**하는 것이다.
    
    당신의 최종 분석은 다음을 반드시 포함해야 한다:
    1.  **최종 결정:** 어떤 전략을 선택해야 하는가?
    2.  **결정 이유:** 왜 그 전략이 최선인가? (3대 공리 및 30년차 전문가 통찰 기반)
    3.  **초기 3개월 로드맵:** 선택된 전략을 실행하기 위한 가장 중요한 첫 3개월간의 구체적인 실행 계획과 핵심 KPI는 무엇인가?
    
    **중요: 이 '최종장' 섹션은 반드시, 처음부터 끝까지 완벽한 한국어로 작성되어야 한다.**
    """
    user_prompt = f"Original User Directive: \"{original_prompt}\"\n\nPreceding Reports:\n{multi_layered_report_structure}\n\nSynthesize the final 'Quantum Oracle Analysis and Genesis Command' section to complete the report."

    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}", "Content-Type": "application/json"}
    payload = {"model": OPENAI_MODEL, "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}], "temperature": 0.1}
    r = await post_with_retries(client, "GPT-Orchestrator", url, headers=headers, json=payload)
    
    synthesis = safe_get(r.json(), ["choices", 0, "message", "content"], default="[최종 종합 분석 중 오류 발생]")
    
    return multi_layered_report_structure + synthesis

# ================== Main Route ==================
@app.post("/route", response_model=RouteOut, tags=["ZINO-GE Core v17.0 Insight Engine"])
async def route(
    payload: RouteIn,
    request: Request,
    x_internal_api_key: Optional[str] = Header(default=None, alias="X-Internal-API-Key"),
):
    if INTERNAL_API_KEY and x_internal_api_key != INTERNAL_API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized: Invalid internal API key")
    
    if not all([os.getenv("OPENAI_API_KEY"), os.getenv("ANTHROPIC_API_KEY"), os.getenv("GEMINI_API_KEY")]):
        return RouteOut(
            report_md="## 시스템 오류\n필수 API 키 일부가 설정되지 않았습니다.",
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
            f"**Gemini:**\n```\n{gemini_res[:1000]}...\n```\n"
            f"**Claude:**\n```\n{claude_res[:1000]}...\n```\n"
            f"**GPT-Creative:**\n```\n{gpt_res[:1000]}...\n```"
        )

    meta_data = {
        "gemini_report": gemini_res,
        "claude_report": claude_res,
        "gpt_creative_report": gpt_res,
    }
    return RouteOut(report_md=final_report, meta=meta_data)
