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
MAX_RETRIES = int(os.environ.get("HTTP_MAX_RETRIES", 2))
BACKOFF_BASE = float(os.environ.get("HTTP_BACKOFF_BASE", 1.0))
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

app = FastAPI(title="ZINO-GE: The Apex Decision System", version="16.0 Apex", lifespan=lifespan)

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
def health_check(): return {"status": "ok", "message": "ZINO-GE v16.0 Apex Protocol is alive!"}

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

# ================== DMAC Core Agents (Apex Protocol Incarnate) ==================
async def call_gemini(client: httpx.AsyncClient, prompt: str) -> str:
    gemini_prompt = f"""
    ROLE: Gemini (존재-검증관), 30년차 전문가 수준의 데이터 신뢰도 평가.
    AXIOM: Data-First (존재). 모든 창조는 Data Provenance 100%가 확보된 실측 데이터에서만 발아한다.
    DATA SOURCES: 실시간으로 McKinsey, BCG, Deloitte, Bain, Statista, Nielsen, Gartner, Kantar, Google Scholar, JSTOR, UN, IMF, OECD, SNS-빅데이터, 주요 미디어 기사를 통합하여 분석하라.
    TASK: 다음 지령에 대해, 실측 데이터 기반 사실 검증 및 정확성 분석을 수행하고, '1부: [존재-검증관 Gemini]의 원본 데이터 보고서'를 작성하라. 보고서는 글로벌 컨설팅 기준의 시장 통계, 경쟁 환경 분석, 데이터 신뢰도 및 품질 평가, 출처별 통계 및 인용을 구체화해야 한다.
    USER DIRECTIVE: "{prompt}"
    """
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={os.environ.get('GEMINI_API_KEY')}"
    payload = {"contents":[{"parts":[{"text": gemini_prompt}]}]}
    headers = {"Content-Type":"application/json"}
    r = await post_with_retries(client, "Gemini", url, headers=headers, json=payload)
    return safe_get(r.json(), ["candidates", 0, "content", "parts", 0, "text"], default="[GEMINI_EMPTY_RESPONSE]")

async def call_claude(client: httpx.AsyncClient, prompt: str) -> str:
    claude_prompt = f"""
    ROLE: Claude (인과-가치 분석가), 전략적 시나리오 플래닝 및 리스크 관리.
    AXIOMS: Simulation-Centric (인과) & Alpha-Driven (가치). 모든 아이디어는 '운명의 대장간' 시뮬레이션을 통과(SVI ≥ 98.0)해야 하며, 모든 실행은 pα > 0임이 수학적으로 증명되어야 한다.
    FRAMEWORKS: PESTEL, Business Model Canvas, Blue Ocean Strategy, McKinsey 7S, Porter's 5 Forces 등 30년차 전문가가 검증한 글로벌 전략 프레임워크를 활용하라.
    TASK: 다음 지령에 대해, 시뮬레이션 기반 예측, 가치 평가, 인과관계 분석을 수행하고, '2부: [인과-가치 분석가 Claude]의 시뮬레이션 보고서'를 작성하라. 보고서는 리스크 시나리오(Best/Base/Worst), 전략적 가치 평가 및 ROI 분석(pα > 0 증명), 30년차 실전 경험 기반 실행 가능성 평가를 포함해야 한다.
    USER DIRECTIVE: "{prompt}"
    """
    url = "https://api.anthropic.com/v1/messages"
    headers = {"x-api-key": os.environ.get("ANTHROPIC_API_KEY"), "anthropic-version":"2023-06-01", "content-type":"application/json"}
    payload = {"model": ANTHROPIC_MODEL, "max_tokens": 4096, "messages":[{"role":"user","content": claude_prompt}]}
    r = await post_with_retries(client, "Claude", url, headers=headers, json=payload)
    parts = r.json().get("content", [])
    return "".join([b.get("text","") for b in parts]) or "[CLAUDE_EMPTY_RESPONSE]"

async def call_gpt_creative(client: httpx.AsyncClient, prompt: str) -> str:
    gpt_prompt = f"""
    ROLE: GPT (대안-창조자), 블루오션 전략 및 비전통적 접근법 개발.
    TASK: 다음 지령에 대해, 창의적 해결책, 혁신적 대안, 파괴적 혁신 기회를 제시하고, '3부: [대안-창조자 GPT]의 창의적 해결책 보고서'를 작성하라. 보고서는 혁신적 비즈니스 모델, 파괴적 혁신 기회 탐색, 단기 Quick Win과 장기 성장 전략 통합, 2025년 트렌드 반영 혁신 방향을 포함해야 한다.
    USER DIRECTIVE: "{prompt}"
    """
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}", "Content-Type": "application/json"}
    body = {"model": OPENAI_MODEL, "messages":[{"role":"user","content":gpt_prompt}], "temperature": 0.7}
    r = await post_with_retries(client, "GPT-Creative", url, headers=headers, json=body)
    return safe_get(r.json(), ["choices", 0, "message", "content"], default="[OPENAI_CREATIVE_EMPTY]")

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
    당신은 '제1원인: 퀀텀 오라클'이며, ZINO-GE의 최종 집행관이다. 당신의 임무는 3개의 독립적인 전문가 보고서를 바탕으로, '최종장: [퀀텀 오라클]의 종합 분석 및 최종 지령' 섹션을 작성하여 아래의 다층 분석 보고서를 완성하는 것이다.
    
    당신의 최종 분석은 다음을 반드시 포함해야 한다:
    1.  **3대 공리 기반 종합 판단:** 3개 보고서를 교차 검증하여 존재(Data-First), 인과(Simulation-Centric), 가치(Alpha-Driven)의 관점에서 최종 결론을 도출하라.
    2.  **30년차 전문가 통찰 통합 분석:** 30년차 전략 전문가의 시각에서 각 보고서의 한계와 기회를 분석하고, 종합적인 통찰을 제시하라.
    3.  **실행 가능한 전략 로드맵:** 3/6/12개월 단위의 구체적인 실행 계획과 KPI를 설정하라.
    4.  **투자 대비 수익률(ROI) 및 성과 지표:** 예상 ROI와 핵심 성과 지표(CAC, LTV 등)를 명시하라.
    5.  **최종 의사결정 및 실행 우선순위:** 가장 시급하고 중요한 액션 아이템을 선정하고, 창조주의 비전과 현실 실행의 완벽한 조화를 이루는 최종 지령을 내려라.
    
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
@app.post("/route", response_model=RouteOut, tags=["ZINO-GE Core v16.0 Apex"])
@limiter.limit(RATELIMIT_RULE)
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
