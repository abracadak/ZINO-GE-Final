import os
import asyncio
import httpx
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import structlog

# --- Logging Setup ---
try:
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
ENABLE_RATELIMIT = os.environ.get("ENABLE_RATELIMIT", "true").lower() == "true"

# ================== FastAPI App & Lifespan ==================
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.http = httpx.AsyncClient(timeout=TIMEOUT_SEC, limits=httpx.Limits(max_connections=100))
    yield
    await app.state.http.aclose()

app = FastAPI(title="ZINO-GE: The Apex Decision System", version="18.0 Live Engine", lifespan=lifespan)

# ================== Middleware ==================
app.add_middleware(CORSMiddleware, allow_origins=CORS_ALLOWED.split(",") if CORS_ALLOWED else ["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# ================== API Schemas & Health Check ==================
class RouteIn(BaseModel):
    user_input: str

class RouteOut(BaseModel):
    report_md: str
    meta: dict

@app.get("/", tags=["Health Check"])
def health_check():
    return {"status": "ok", "message": "ZINO-GE v18.0 Live Engine is alive!"}

# ================== Utility Functions ==================
def safe_get(d: dict, path: list, default: Any = "") -> Any:
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

# ================== Core Agents (AI and Data Collection) ==================
async def fetch_realtime_data(client: httpx.AsyncClient, query: str) -> str:
    """
    사용자 쿼리와 관련된 최신 뉴스 및 데이터를 웹에서 검색합니다.
    (현재는 시뮬레이션된 데이터를 반환하며, 실제 구현 시에는 NewsAPI, Google Search API 등을 연동합니다.)
    """
    log.info("fetching_realtime_data", query=query)
    await asyncio.sleep(0.5)  # 네트워크 지연 시뮬레이션
    return f"시뮬레이션된 실시간 데이터: '{query}'와(과) 관련된 최신 시장 보고서에 따르면, 관련 기술의 잠재적 시장 가치는 향후 5년간 25% 성장할 것으로 예상됩니다."

async def call_gemini(client: httpx.AsyncClient, prompt: str, real_time_context: str) -> str:
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

# ================== Main Route ==================
@app.post("/route", response_model=RouteOut, tags=["ZINO-GE Core v18.0 Live Engine"])
async def route(payload: RouteIn, request: Request, x_internal_api_key: Optional[str] = Header(default=None, alias="X-Internal-API-Key")):
    if INTERNAL_API_KEY and x_internal_api_key != INTERNAL_API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized: Invalid internal API key")
    
    if not all([os.getenv("OPENAI_API_KEY"), os.getenv("ANTHROPIC_API_KEY"), os.getenv("GEMINI_API_KEY")]):
        return RouteOut(report_md="## 시스템 오류\n필수 API 키 일부가 설정되지 않았습니다.", meta={"error":"SERVER_CONFIG_MISSING_KEYS"})

    client: httpx.AsyncClient = request.app.state.http

    # 1단계: 실시간 데이터 수집
    real_time_data = await fetch_realtime_data(client, payload.user_input)

    # 2단계: AI 엔진 호출 및 결과 수집
    tasks = [
        call_gemini(client, payload.user_input, real_time_data),
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

    # 3단계: 최종 보고서 생성
    try:
        final_report = await call_gpt_orchestrator(client, payload.user_input, [gemini_res, claude_res, gpt_res])
    except Exception as e:
        log.exception("orchestration_failed", error=str(e))
        final_report = f"## 최종 종합 실패\n- **오류 원인:** {type(e).__name__}"

    meta_data = {
        "gemini_report": gemini_res,
        "claude_report": claude_res,
        "gpt_creative_report": gpt_res,
    }
    
    return RouteOut(report_md=final_report, meta=meta_data)
