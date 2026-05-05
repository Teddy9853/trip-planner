import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from agents.orchestrator import OrchestratorAgent
from mcp.server import MCPServer
from rag.engine import RAGEngine

# Optional MCP support
try:
    from fastapi_mcp import FastApiMCP
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False

load_dotenv()

logger = logging.getLogger(__name__)

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY is missing. Add it to your .env file.")

# ---------------------------------------------------------------------------
# Application-level singletons (initialised in lifespan)
# ---------------------------------------------------------------------------

_mcp_server: Optional[MCPServer] = None
_rag_engine: Optional[RAGEngine] = None
_orchestrator: Optional[OrchestratorAgent] = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """Initialise shared resources on startup and clean up on shutdown."""
    global _mcp_server, _rag_engine, _orchestrator

    _mcp_server = MCPServer()
    _rag_engine = RAGEngine(_mcp_server)
    await _rag_engine.initialize()
    _orchestrator = OrchestratorAgent(_rag_engine, _mcp_server)
    logger.info("Application startup complete — RAGEngine and OrchestratorAgent ready.")

    yield

    # Shutdown: close the httpx client inside MCPServer
    if _mcp_server is not None:
        await _mcp_server.__aexit__(None, None, None)
    logger.info("Application shutdown complete.")


app = FastAPI(title="AI Trip Planner", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class TripRequest(BaseModel):
    origin: str = Field(..., json_schema_extra={"example": "New York"})
    destination: str = Field(..., json_schema_extra={"example": "Tokyo"})
    travelers: int = Field(..., ge=1, json_schema_extra={"example": 2})
    budget_usd: Optional[int] = Field(None, ge=0, json_schema_extra={"example": 3000})
    days: Optional[int] = Field(None, ge=1, le=30, json_schema_extra={"example": 5})
    interests: List[str] = Field(
        default_factory=list,
        json_schema_extra={"example": ["food", "anime", "culture"]},
    )


class Stop(BaseModel):
    """Enhanced Stop model with full Attraction_Card fields (requirement 8.1)."""

    # Core fields
    name: str
    day: int
    description: str
    lat: float = Field(..., ge=-90.0, le=90.0)
    lng: float = Field(..., ge=-180.0, le=180.0)

    # Enhanced fields (task 2.1)
    time: str = Field(default="09:00", pattern=r"^\d{2}:\d{2}$")
    rating: float = Field(default=0.0, ge=0.0, le=5.0)
    review_count: int = Field(default=0, ge=0)
    category: str = Field(default="景點")
    ai_notes: str = Field(default="")
    website_url: Optional[str] = Field(default=None)
    image_urls: List[str] = Field(default_factory=list, max_length=5)


class BudgetAnalysis(BaseModel):
    """Budget breakdown produced by Budget_Agent."""

    transportation: float = Field(default=0.0, ge=0.0, description="交通費用估算（USD）")
    accommodation: float = Field(default=0.0, ge=0.0, description="住宿費用估算（USD）")
    food: float = Field(default=0.0, ge=0.0, description="餐飲費用估算（USD）")
    activities: float = Field(default=0.0, ge=0.0, description="活動費用估算（USD）")
    total: float = Field(default=0.0, ge=0.0, description="總預算估算（USD）")
    per_person: float = Field(default=0.0, ge=0.0, description="每人費用估算（USD）")
    notes: str = Field(default="", description="預算備註與建議")


class TransportSuggestions(BaseModel):
    """Daily transport suggestions produced by Transport_Agent."""

    daily_suggestions: List[dict] = Field(
        default_factory=list,
        description="每日交通建議清單，每項包含 day、mode、description、estimated_time",
    )
    general_tips: str = Field(default="", description="整體交通建議")


class ProgressEvent(BaseModel):
    """SSE progress event pushed to the frontend during trip planning."""

    event: str = Field(..., description="事件類型：progress | complete | error")
    message: str = Field(..., description="人類可讀的進度訊息")
    data: Optional[dict] = Field(default=None, description="complete 時包含完整 TripResponse")


class TripResponse(BaseModel):
    origin: str
    destination: str
    travelers: int
    budget_usd: Optional[int]
    days: Optional[int]
    plan: str
    stops: List[Stop]
    budget_analysis: Optional[BudgetAnalysis] = None
    transport_suggestions: Optional[TransportSuggestions] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_day_color(day: int) -> str:
    colors = [
        "red", "blue", "green", "orange", "purple",
        "yellow", "violet", "grey", "black",
    ]
    return colors[(day - 1) % len(colors)]


async def _validate_location(location: str) -> None:
    """
    Validate that *location* can be geocoded by Nominatim.

    Raises HTTPException 422 if the location cannot be recognised
    (requirement 10.5).
    """
    if _mcp_server is None:
        return  # skip validation if server not ready (e.g. during tests)
    try:
        await _mcp_server.nominatim_geocode(query=location)
    except Exception:
        raise HTTPException(
            status_code=422,
            detail=f"無法識別地點：{location}，請確認拼寫是否正確",
        )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/", include_in_schema=False)
def home():
    return FileResponse("ui.html")


@app.get("/health", operation_id="health_check")
def health():
    return {"status": "ok"}


@app.post("/plan-trip", response_model=TripResponse, operation_id="plan_trip")
async def plan_trip(req: TripRequest):
    """
    Synchronous trip planning endpoint.

    Replaces the direct OpenAI call with OrchestratorAgent.plan().
    Adds:
      - Location geocoding validation (requirement 10.5)
      - 60-second timeout with HTTP 408 on expiry (requirement 9.3)

    Requirements: 8.3, 9.1, 9.3
    """
    # Task 7.3 — validate origin and destination (requirement 10.5)
    await _validate_location(req.origin)
    await _validate_location(req.destination)

    if _orchestrator is None:
        raise HTTPException(status_code=503, detail="Service not ready. Please retry.")

    try:
        # Task 7.1 — delegate to OrchestratorAgent with 60-second timeout
        response: TripResponse = await asyncio.wait_for(
            _orchestrator.plan(req, progress_callback=None),
            timeout=60.0,
        )
        return response

    except asyncio.TimeoutError:
        # Requirement 9.3 — return HTTP 408 with TIMEOUT_ERROR
        raise HTTPException(
            status_code=408,
            detail={"error_code": "TIMEOUT_ERROR", "message": "行程產生逾時，請重試"},
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("plan_trip failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/plan-trip/stream", operation_id="plan_trip_stream")
async def plan_trip_stream(
    origin: str = Query(..., description="出發地"),
    destination: str = Query(..., description="目的地"),
    travelers: int = Query(..., ge=1, description="旅客人數"),
    budget_usd: Optional[int] = Query(None, ge=0, description="預算（USD）"),
    days: Optional[int] = Query(None, ge=1, le=30, description="行程天數"),
    interests: List[str] = Query(default=[], description="興趣偏好"),
):
    """
    SSE streaming trip planning endpoint (requirement 9.2).

    Accepts the same parameters as POST /plan-trip via query string.
    Pushes:
      - ``progress`` events while planning is in progress
      - ``complete`` event with the full TripResponse on success
      - ``error`` event on failure

    Requirements: 9.2
    """
    req = TripRequest(
        origin=origin,
        destination=destination,
        travelers=travelers,
        budget_usd=budget_usd,
        days=days,
        interests=interests,
    )

    async def event_generator() -> AsyncGenerator:
        # Task 7.3 — validate locations before starting the stream
        try:
            await _validate_location(req.origin)
            await _validate_location(req.destination)
        except HTTPException as exc:
            yield {
                "event": "error",
                "data": json.dumps(
                    {"error_code": "INVALID_LOCATION", "message": exc.detail},
                    ensure_ascii=False,
                ),
            }
            return

        if _orchestrator is None:
            yield {
                "event": "error",
                "data": json.dumps(
                    {"error_code": "SERVICE_UNAVAILABLE", "message": "Service not ready. Please retry."},
                    ensure_ascii=False,
                ),
            }
            return

        # Collect progress messages from the orchestrator and forward as SSE
        progress_queue: asyncio.Queue[str] = asyncio.Queue()

        async def progress_callback(message: str) -> None:
            await progress_queue.put(message)

        # Run the orchestrator in a background task so we can interleave
        # progress events while it executes.
        planning_task = asyncio.create_task(
            _orchestrator.plan(req, progress_callback=progress_callback)
        )

        try:
            while not planning_task.done():
                # Drain any queued progress messages
                while not progress_queue.empty():
                    msg = progress_queue.get_nowait()
                    yield {
                        "event": "progress",
                        "data": json.dumps({"message": msg}, ensure_ascii=False),
                    }
                # Yield control briefly so the planning task can make progress
                await asyncio.sleep(0.1)

            # Drain any remaining progress messages after task completion
            while not progress_queue.empty():
                msg = progress_queue.get_nowait()
                yield {
                    "event": "progress",
                    "data": json.dumps({"message": msg}, ensure_ascii=False),
                }

            # Retrieve the result (re-raises any exception from the task)
            result: TripResponse = planning_task.result()

            yield {
                "event": "complete",
                "data": json.dumps(result.model_dump(), ensure_ascii=False),
            }

        except asyncio.TimeoutError:
            planning_task.cancel()
            yield {
                "event": "error",
                "data": json.dumps(
                    {"error_code": "TIMEOUT_ERROR", "message": "行程產生逾時，請重試"},
                    ensure_ascii=False,
                ),
            }
        except Exception as exc:
            logger.exception("plan_trip_stream failed: %s", exc)
            yield {
                "event": "error",
                "data": json.dumps(
                    {"error_code": "INTERNAL_ERROR", "message": str(exc)},
                    ensure_ascii=False,
                ),
            }

    return EventSourceResponse(event_generator())


if MCP_AVAILABLE:
    mcp = FastApiMCP(app)
    mcp.mount()
