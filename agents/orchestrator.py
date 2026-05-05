"""
Orchestrator Agent — coordinates all sub-agents to produce a TripResponse.

Execution flow (requirements 7.1, 7.6, 7.7, 7.8):

  Step 1 (sequential):
    ItineraryAgent.generate_draft()
    → progress_callback("Generating itinerary draft...")

  Step 2 (parallel via asyncio.gather):
    AttractionAgent.enrich_all()   ─┐
    BudgetAgent.analyze()           ├─ asyncio.gather
    TransportAgent.suggest()       ─┘
    → progress_callback("Fetching attraction details...")

  Step 3 (sequential):
    Assemble TripResponse from all results
    → progress_callback("Assembling final response...")
    → return TripResponse

Degradation (requirement 7.7, 10.4):
  Any sub-agent exception is caught, logged with timestamp/error_type/message,
  and execution continues with the remaining agents.  The final TripResponse
  may have None for budget_analysis or transport_suggestions if those agents
  failed, but stops are always present (or empty on total failure).
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Callable, List, Optional

from agents.attraction import AttractionAgent, EnrichedStop
from agents.budget import BudgetAgent
from agents.itinerary import ItineraryAgent
from agents.transport import TransportAgent

if TYPE_CHECKING:
    from mcp.server import MCPServer
    from rag.engine import RAGEngine
    from server import BudgetAnalysis, TransportSuggestions, TripRequest, TripResponse

logger = logging.getLogger(__name__)


class OrchestratorAgent:
    """
    Coordinates all sub-agents to produce a complete TripResponse.

    Usage::

        orchestrator = OrchestratorAgent(rag_engine, mcp_server)
        response = await orchestrator.plan(request, progress_callback)

    The *progress_callback* is an async callable that accepts a single string
    message.  It is used to push SSE progress events to the frontend.
    """

    def __init__(self, rag_engine: "RAGEngine", mcp_server: "MCPServer") -> None:
        self._rag = rag_engine
        self._mcp = mcp_server

        # Instantiate sub-agents
        self._itinerary_agent = ItineraryAgent()
        self._attraction_agent = AttractionAgent(rag_engine, mcp_server)
        self._budget_agent = BudgetAgent()
        self._transport_agent = TransportAgent()

    # ------------------------------------------------------------------
    # Error logging helper (requirement 10.4)
    # ------------------------------------------------------------------

    def _log_agent_error(self, agent_name: str, exc: Exception) -> None:
        """Log a structured error entry for a failed sub-agent (requirement 7.7, 10.4)."""
        logger.error(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "error_type": type(exc).__name__,
                "agent": agent_name,
                "error_message": str(exc),
            }
        )

    # ------------------------------------------------------------------
    # Main plan method (requirements 7.1, 7.6, 7.8)
    # ------------------------------------------------------------------

    async def plan(
        self,
        request: "TripRequest",
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> "TripResponse":
        """
        Orchestrate all sub-agents and return a complete TripResponse.

        *progress_callback* is called with human-readable status strings so
        the SSE endpoint can forward them to the frontend.  It may be a plain
        function or a coroutine function; both are handled.

        Requirements: 7.1, 7.6, 7.7, 7.8
        """
        # Import here to avoid circular imports at module load time
        from server import Stop, TripResponse

        async def _notify(message: str) -> None:
            if progress_callback is None:
                return
            try:
                result = progress_callback(message)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as exc:
                logger.warning("progress_callback raised: %s", exc)

        # ----------------------------------------------------------------
        # Step 1 — Generate itinerary draft
        # ----------------------------------------------------------------
        await _notify("Generating itinerary draft...")
        draft = None
        try:
            draft = await self._itinerary_agent.generate_draft(request)
            logger.info(
                "ItineraryAgent produced %d stops over %d days",
                len(draft.stops),
                draft.days,
            )
        except Exception as exc:
            self._log_agent_error("ItineraryAgent", exc)
            # Cannot continue without a draft — return minimal response
            logger.error("ItineraryAgent failed fatally; returning empty response")
            return TripResponse(
                origin=request.origin,
                destination=request.destination,
                travelers=request.travelers,
                budget_usd=request.budget_usd,
                days=request.days,
                plan="(Itinerary draft generation failed, please retry)",
                stops=[],
            )

        # ----------------------------------------------------------------
        # Step 2 — Parallel enrichment, budget analysis, transport
        # ----------------------------------------------------------------
        await _notify("Fetching attraction details...")

        enriched_stops: List[EnrichedStop] = []
        budget_analysis: Optional["BudgetAnalysis"] = None
        transport_suggestions: Optional["TransportSuggestions"] = None

        async def _enrich_all():
            return await self._attraction_agent.enrich_all(draft.stops)

        async def _analyze_budget():
            return await self._budget_agent.analyze(draft, request)

        async def _suggest_transport():
            return await self._transport_agent.suggest(draft)

        # Run all three in parallel; capture exceptions individually
        results = await asyncio.gather(
            _enrich_all(),
            _analyze_budget(),
            _suggest_transport(),
            return_exceptions=True,
        )

        enrich_result, budget_result, transport_result = results

        # Attraction enrichment
        if isinstance(enrich_result, Exception):
            self._log_agent_error("AttractionAgent", enrich_result)
            logger.warning("AttractionAgent failed — stops will use draft data only")
            # Fall back to minimal Stop objects from the draft
            enriched_stops = []
        else:
            enriched_stops = enrich_result

        # Budget analysis
        if isinstance(budget_result, Exception):
            self._log_agent_error("BudgetAgent", budget_result)
            logger.warning("BudgetAgent failed — budget_analysis will be None")
            budget_analysis = None
        else:
            budget_analysis = budget_result

        # Transport suggestions
        if isinstance(transport_result, Exception):
            self._log_agent_error("TransportAgent", transport_result)
            logger.warning("TransportAgent failed — transport_suggestions will be None")
            transport_suggestions = None
        else:
            transport_suggestions = transport_result

        # ----------------------------------------------------------------
        # Step 3 — Assemble final TripResponse
        # ----------------------------------------------------------------
        await _notify("Assembling final response...")

        # Convert EnrichedStop → Stop (the API response model)
        stops: List[Stop] = []
        if enriched_stops:
            for es in enriched_stops:
                stops.append(
                    Stop(
                        name=es.name,
                        day=es.day,
                        time=es.time,
                        description=es.description,
                        lat=es.lat,
                        lng=es.lng,
                        rating=es.rating,
                        review_count=es.review_count,
                        category=es.category,
                        ai_notes=es.ai_notes,
                        website_url=es.website_url,
                        image_urls=es.image_urls[:5],
                    )
                )
        else:
            # AttractionAgent failed — build minimal stops from draft
            for ds in draft.stops:
                stops.append(
                    Stop(
                        name=ds.name,
                        day=ds.day,
                        time=ds.time,
                        description=ds.description,
                        lat=ds.lat if ds.lat is not None else 0.0,
                        lng=ds.lng if ds.lng is not None else 0.0,
                        ai_notes="(Source: AI-generated) (Coordinates are AI-estimated)",
                    )
                )

        response = TripResponse(
            origin=request.origin,
            destination=request.destination,
            travelers=request.travelers,
            budget_usd=request.budget_usd,
            days=draft.days,
            plan=draft.plan_text,
            stops=stops,
            budget_analysis=budget_analysis,
            transport_suggestions=transport_suggestions,
        )

        logger.info(
            "OrchestratorAgent completed: %d stops, budget=%s, transport=%s",
            len(stops),
            "ok" if budget_analysis else "degraded",
            "ok" if transport_suggestions else "degraded",
        )

        return response
