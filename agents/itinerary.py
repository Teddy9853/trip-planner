"""
Itinerary Agent — generates a daily itinerary draft via OpenAI GPT-4.1-mini.

Produces an ItineraryDraft containing a list of DraftStop objects with basic
attraction information (name, day, time, description, suggested duration).
The Attraction_Agent will later enrich each DraftStop with detailed data.

Requirement: 7.2
"""

from __future__ import annotations

import json
import logging
import os
from typing import List, Optional

from openai import AsyncOpenAI
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models shared across the agent layer
# ---------------------------------------------------------------------------


class DraftStop(BaseModel):
    """A basic stop produced by the Itinerary_Agent before enrichment."""

    name: str = Field(..., description="景點名稱")
    day: int = Field(..., ge=1, description="行程天數")
    time: str = Field(default="09:00", pattern=r"^\d{2}:\d{2}$", description="建議到達時間（HH:MM）")
    description: str = Field(default="", description="活動描述")
    suggested_duration_hours: float = Field(default=1.0, ge=0.0, description="建議停留時間（小時）")
    city: str = Field(default="", description="所在城市（用於 RAG 查詢）")
    country: str = Field(default="", description="所在國家（用於 RAG 查詢）")
    # Rough coordinates from LLM — may be refined by Nominatim later
    lat: Optional[float] = Field(default=None, description="緯度（LLM 估算）")
    lng: Optional[float] = Field(default=None, description="經度（LLM 估算）")


class ItineraryDraft(BaseModel):
    """Complete itinerary draft returned by ItineraryAgent.generate_draft()."""

    origin: str = Field(..., description="出發地")
    destination: str = Field(..., description="目的地")
    days: int = Field(..., ge=1, description="行程天數")
    plan_text: str = Field(default="", description="人類可讀的行程說明文字")
    stops: List[DraftStop] = Field(default_factory=list, description="每日景點清單")


# ---------------------------------------------------------------------------
# Itinerary Agent
# ---------------------------------------------------------------------------


class ItineraryAgent:
    """
    Calls OpenAI GPT-4.1-mini to produce a structured daily itinerary draft.

    Usage::

        agent = ItineraryAgent()
        draft = await agent.generate_draft(request)
    """

    def __init__(self) -> None:
        self._openai = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    async def generate_draft(self, request) -> ItineraryDraft:
        """
        Generate a daily itinerary draft for the given TripRequest.

        The LLM is asked to return structured JSON with a plan text and a
        list of stops.  Each stop includes the attraction name, day, time,
        description, suggested duration, and rough coordinates.

        Requirement: 7.2
        """
        budget_text = (
            f"Budget is ${request.budget_usd} USD total."
            if request.budget_usd is not None
            else "No strict budget limit."
        )

        days_text = (
            f"Trip duration is {request.days} days."
            if request.days is not None
            else "Trip duration is flexible (suggest 3–5 days)."
        )

        interests_text = (
            ", ".join(request.interests)
            if request.interests
            else "general travel, food, culture, sightseeing"
        )

        # Infer number of days for the schema
        num_days = request.days if request.days is not None else 3

        prompt = f"""You are an expert travel planner.

Create a detailed day-by-day itinerary for the following trip:

- Origin: {request.origin}
- Destination: {request.destination}
- Travelers: {request.travelers}
- {budget_text}
- {days_text}
- Interests: {interests_text}

Return a JSON object with:
1. "plan_text": a comprehensive human-readable travel plan (in Traditional Chinese, 繁體中文)
2. "stops": an array of attraction stops

Each stop must include:
- "name": attraction name (in local language or English)
- "day": day number (1 to {num_days})
- "time": suggested arrival time in HH:MM format (24-hour)
- "description": brief activity description (in Traditional Chinese)
- "suggested_duration_hours": recommended time to spend (decimal hours, e.g. 1.5)
- "city": city where the attraction is located
- "country": country where the attraction is located
- "lat": approximate latitude (decimal degrees)
- "lng": approximate longitude (decimal degrees)

Rules:
- Include 3–5 stops per day
- Arrange stops in logical geographic order within each day
- Spread stops across all {num_days} days
- Include the origin as day 1 departure point if relevant
- Provide realistic coordinates for each stop
"""

        response = await self._openai.responses.create(
            model="gpt-4.1-mini",
            input=prompt,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "itinerary_draft",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "plan_text": {"type": "string"},
                            "stops": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "name": {"type": "string"},
                                        "day": {"type": "integer", "minimum": 1},
                                        "time": {"type": "string"},
                                        "description": {"type": "string"},
                                        "suggested_duration_hours": {"type": "number", "minimum": 0},
                                        "city": {"type": "string"},
                                        "country": {"type": "string"},
                                        "lat": {"type": "number", "minimum": -90, "maximum": 90},
                                        "lng": {"type": "number", "minimum": -180, "maximum": 180},
                                    },
                                    "required": [
                                        "name", "day", "time", "description",
                                        "suggested_duration_hours", "city", "country",
                                        "lat", "lng",
                                    ],
                                    "additionalProperties": False,
                                },
                            },
                        },
                        "required": ["plan_text", "stops"],
                        "additionalProperties": False,
                    },
                    "strict": True,
                }
            },
        )

        data = json.loads(response.output_text)

        stops = [DraftStop(**s) for s in data.get("stops", [])]
        actual_days = max((s.day for s in stops), default=num_days)

        logger.info(
            "ItineraryAgent generated %d stops over %d days for %s → %s",
            len(stops),
            actual_days,
            request.origin,
            request.destination,
        )

        return ItineraryDraft(
            origin=request.origin,
            destination=request.destination,
            days=actual_days,
            plan_text=data.get("plan_text", ""),
            stops=stops,
        )
