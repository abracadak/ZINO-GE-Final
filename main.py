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

app = FastAPI(title="ZINO-GE: The Apex Decision System", version="18.0 Live Engine", lifespan=lifespan)

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
class RouteIn(BaseModel):
    user_input: str

class RouteOut(BaseModel):
    report_md: str
    meta: Dict[str, Any]

@app.get("/", tags=["Health Check"])
def health_check():
    return {"status": "ok", "message": "ZINO-GE v18.0 Live Engine is alive!"}

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

# ================== DMAC Core Agents (Live Data Protocol) ==================

# ✅ 신규 추가: 실시간 데이터 수집 에이전트
async def fetch_realtime_data(client: httpx.AsyncClient, query: str) -> str:
    """
    사용자 쿼리와 관련된 최신 뉴스 및 데이터를 웹에서 검색합니다.
    (현재는 시뮬레이션된 데이터를 반환하며, 실제 구현 시에는 NewsAPI, Google Search API 등을 연동합니다.)
    """
    log.info("fetching_realtime_data", query=query)
    # 실제 구현 예시:
    # try:
    #     news_api_key = os.environ.get("NEWS_API_KEY")
    #     search_url = f"https://newsapi.org/v2/everything?q={query}&apiKey={news_api_key}"
    #     response = await client.get(search_url)
    #     response.raise_for_status()
    #     articles = response.json().get("articles", [])
    #     snippets = [f"Title: {a['title']}\nSnippet: {a['description']}" for a in articles[:3]]
    #     return "\n---\n".join(snippets) if snippets else "No real-time news found."
    # except Exception as e:
    #     log.error("realtime_data_fetch_failed", error=str(e))
    #     return f"Failed to fetch real-time data: {e}"
    
    # 현재는 안정적인 테스트를 위해 시뮬레이션된 데이터를 반환합니다.
    await asyncio.sleep(0.5) # 네트워크 지연 시뮬레이션
    return f"시뮬레이션된 실시간 데이터: '{query}'와(과) 관련된 최신 시장 보고서에 따르면, 관련 기술의 잠재적 시장 가치는 향후 5년간 25% 성장할 것으로 예상됩니다."

async def call_gemini(client: httpx.AsyncClient, prompt: str, real_time_context: str) -> str:
    # ✅ 수정: 실시간 데이터를 프롬프트에 포함
    gemini_prompt = f"""
    ROLE: Gemini (존재-검증관). 당신은 30년차 수석 애널리스트다.
    AXIOM: Data-First (존재).
    PROVIDED REAL-TIME DATA: 아래 제공된 실시간 데이터를 최우선 분석 소스로 사용하라.
    ---
    {real_time_context}
    ---
    TASK: 제공된 실시간 데이터와 기존 지식을 종합하여, 사용자 지령에 대한 데이터의 '결핍'과 '편향'을 분석하라. 세상이 무엇을 알고 모르는지를 명확히 하고, 데이터의 신뢰성을 평가하여 보고서를 작성하라.
    USER DIRECTIVE: "{prompt}"
    """
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={os.environ.get('GEMINI_API_KEY')}"
    payload = {"contents":[{"parts":[{"text": gemini_prompt}]}]}
    headers = {"Content-Type":"application/json"}
    r = await post_with_retries(client, "Gemini", url, headers=headers, json=payload)
    return safe_get(r.json(), ["candidates", 0, "content", "parts", 0, "text"], default="[GEMINI_ERROR: No content found]")

async def call_claude(client: httpx.AsyncClient, prompt: str) -> str:
    # ... (내용 동일)
    return "Claude Response"

async def call_gpt_creative(client: httpx.AsyncClient, prompt: str) -> str:
    # ... (내용 동일)
    return "GPT Creative Response"

async def call_gpt_orchestrator(client: httpx.AsyncClient, original_prompt: str, reports: List[str]) -> str:
    # ... (내용 동일)
    return "Final Report"

# ================== Main Route ==================
@app.post("/route", response_model=RouteOut, tags=["ZINO-GE Core v18.0 Live Engine"])
async def route(
    payload: RouteIn,
    request: Request,
    x_internal_api_key: Optional[str] = Header(default=None, alias="X-Internal-API-Key"),
):
    # ... (보안 및 설정 검사는 이전과 동일)

    client: httpx.AsyncClient = request.app.state.http

    # ✅ 수정: 데이터 수집 단계를 추가하고, 그 결과를 Gemini 호출에 사용
    try:
        # 1단계: 실시간 데이터 수집
        real_time_data = await fetch_realtime_data(client, payload.user_input)

        # 2단계: 3명의 전문가에게 동시 지령 하달 (Gemini는 실시간 데이터를 함께 전달)
        tasks = [
            call_gemini(client, payload.user_input, real_time_data),
            call_claude(client, payload.user_input),
            call_gpt_creative(client, payload.user_input),
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # ... (이하 결과 처리 및 최종 종합 로직은 이전과 동일)
        # ... (unwrap 및 최종 리포트 반환)
        gemini_res = "Gemini Result"  # Placeholder
        claude_res = "Claude Result"  # Placeholder
        gpt_res = "GPT Result"  # Placeholder
        final_report = "Final Report"  # Placeholder

    except Exception as e:
        log.exception("critical_error_in_route", error=str(e))
        raise HTTPException(status_code=500, detail="A critical error occurred in the main processing route.")

    meta_data = {
        "real_time_data_summary": real_time_data,  # 메타데이터에 수집된 데이터 요약 추가
        "gemini_report": gemini_res,
        "claude_report": claude_res,
        "gpt_creative_report": gpt_res,
    }
    return RouteOut(report_md=final_report, meta=meta_data)
