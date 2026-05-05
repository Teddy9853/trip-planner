"""
Attraction Agent — enriches DraftStop objects with detailed attraction data.

For each stop it:
  1. Calls RAGEngine.query() to get AI notes, coordinates, and category.
  2. Calls MCPServer.get_attraction_images() to get image URLs.
  3. Combines everything into an EnrichedStop.

Degradation strategy (requirements 10.1, 10.2, 10.3):
  - If RAG fails → use LLM-estimated coordinates, mark ai_notes accordingly.
  - If image fetch fails → image_urls = [] (frontend shows placeholder).
  - Neither failure aborts the overall pipeline.

Requirements: 7.3, 7.8, 10.1, 10.2, 10.3
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, List, Optional

from pydantic import BaseModel, Field

from agents.itinerary import DraftStop

if TYPE_CHECKING:
    from mcp.server import MCPServer
    from rag.engine import RAGEngine

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enriched stop — the fully populated stop ready for the API response
# ---------------------------------------------------------------------------


class EnrichedStop(BaseModel):
    """A DraftStop enriched with RAG data and images."""

    # Core fields (from DraftStop)
    name: str
    day: int
    time: str
    description: str
    city: str = ""
    country: str = ""

    # Coordinates — from Nominatim if available, else LLM estimate
    lat: float = Field(default=0.0, ge=-90.0, le=90.0)
    lng: float = Field(default=0.0, ge=-180.0, le=180.0)

    # Enriched fields
    rating: float = Field(default=0.0, ge=0.0, le=5.0)
    review_count: int = Field(default=0, ge=0)
    category: str = Field(default="Attraction")
    ai_notes: str = Field(default="")
    website_url: Optional[str] = Field(default=None)
    image_urls: List[str] = Field(default_factory=list, max_length=5)

    # Metadata
    coordinates_estimated: bool = Field(
        default=False,
        description="True when coordinates come from LLM estimation, not Nominatim",
    )


# ---------------------------------------------------------------------------
# Attraction Agent
# ---------------------------------------------------------------------------


class AttractionAgent:
    """
    Enriches DraftStop objects with detailed attraction information.

    Usage::

        agent = AttractionAgent(rag_engine, mcp_server)
        enriched = await agent.enrich_stop(draft_stop)
        all_enriched = await agent.enrich_all(draft_stops)
    """

    def __init__(self, rag_engine: "RAGEngine", mcp_server: "MCPServer") -> None:
        self._rag = rag_engine
        self._mcp = mcp_server

    async def enrich_stop(self, stop: DraftStop) -> EnrichedStop:
        """
        Enrich a single DraftStop with RAG data and images.

        Degradation rules:
        - RAG failure → keep LLM coordinates, mark ai_notes with fallback label.
        - Image failure → image_urls = [].
        - Coordinate fallback → set coordinates_estimated = True and append
          "Coordinates are AI-estimated" to ai_notes (requirement 10.1).
        Requirements: 7.3, 10.1, 10.2, 10.3
        """
        # ---- RAG query ------------------------------------------------
        rag_result = None
        rag_failed = False
        try:
            rag_result = await self._rag.query(
                attraction_name=stop.name,
                city=stop.city,
                country=stop.country,
            )
        except Exception as exc:
            logger.error(
                "RAGEngine.query failed for %s: %s — applying degradation",
                stop.name,
                exc,
            )
            rag_failed = True

        # ---- Image fetch ----------------------------------------------
        image_urls: List[str] = []
        try:
            image_urls = await self._mcp.get_attraction_images(
                attraction_name=stop.name, max_images=5
            )
        except Exception as exc:
            logger.warning(
                "get_attraction_images failed for %s: %s — using empty list (requirement 10.2)",
                stop.name,
                exc,
            )
            image_urls = []

        # ---- Assemble EnrichedStop ------------------------------------
        if rag_failed or rag_result is None:
            # Full RAG failure — use LLM draft values (requirement 10.3)
            lat = stop.lat if stop.lat is not None else 0.0
            lng = stop.lng if stop.lng is not None else 0.0
            ai_notes = "(Source: AI-generated) (Coordinates are AI-estimated)"
            category = "Attraction"
            website_url = None
            coordinates_estimated = True
        else:
            # Use Nominatim coordinates when available, else fall back to LLM
            if rag_result.lat is not None and rag_result.lng is not None:
                lat = rag_result.lat
                lng = rag_result.lng
                coordinates_estimated = False
                ai_notes = rag_result.ai_notes
            else:
                lat = stop.lat if stop.lat is not None else 0.0
                lng = stop.lng if stop.lng is not None else 0.0
                coordinates_estimated = True
                # Append coordinate-estimation notice (requirement 10.1)
                ai_notes = rag_result.ai_notes
                if ai_notes and not ai_notes.endswith("(Coordinates are AI-estimated)"):
                    ai_notes += "\n\n(Coordinates are AI-estimated)"
                elif not ai_notes:
                    ai_notes = "(Coordinates are AI-estimated)"

            category = rag_result.category or "Attraction"
            website_url = rag_result.website_url

        enriched = EnrichedStop(
            name=stop.name,
            day=stop.day,
            time=stop.time,
            description=stop.description,
            city=stop.city,
            country=stop.country,
            lat=lat,
            lng=lng,
            rating=rag_result.rating if rag_result else 0.0,
            review_count=rag_result.review_count if rag_result else 0,
            category=category,
            ai_notes=ai_notes,
            website_url=website_url,
            image_urls=image_urls[:5],
            coordinates_estimated=coordinates_estimated,
        )

        logger.info(
            "Enriched stop: %s (day=%d, cache_hit=%s, images=%d)",
            stop.name,
            stop.day,
            rag_result.cache_hit if rag_result else "N/A",
            len(image_urls),
        )

        return enriched

    async def enrich_all(self, stops: List[DraftStop]) -> List[EnrichedStop]:
        """
        Enrich all stops in parallel using asyncio.gather.

        Individual failures are caught inside enrich_stop and result in a
        degraded (but valid) EnrichedStop, so gather never raises.

        Requirement: 7.8
        """
        logger.info("Enriching %d stops in parallel", len(stops))
        results = await asyncio.gather(
            *[self.enrich_stop(stop) for stop in stops],
            return_exceptions=False,  # enrich_stop handles its own exceptions
        )
        return list(results)
