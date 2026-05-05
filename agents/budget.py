"""
Budget Agent — analyses trip costs and returns a BudgetAnalysis.

Calls OpenAI GPT-4.1-mini with the itinerary draft and trip request to
produce a structured cost breakdown (transportation, accommodation, food,
activities) plus a total and per-person estimate.

Requirement: 7.4
"""

from __future__ import annotations

import json
import logging
import os
from typing import TYPE_CHECKING

from openai import AsyncOpenAI

if TYPE_CHECKING:
    from agents.itinerary import ItineraryDraft
    from server import BudgetAnalysis, TripRequest

logger = logging.getLogger(__name__)


class BudgetAgent:
    """
    Analyses trip costs using GPT-4.1-mini.

    Usage::

        agent = BudgetAgent()
        analysis = await agent.analyze(draft, request)
    """

    def __init__(self) -> None:
        self._openai = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    async def analyze(self, draft: "ItineraryDraft", request: "TripRequest") -> "BudgetAnalysis":
        """
        Produce a BudgetAnalysis for the given itinerary draft and trip request.

        Requirement: 7.4
        """
        # Import here to avoid circular imports at module load time
        from server import BudgetAnalysis

        stops_summary = "\n".join(
            f"  Day {s.day}: {s.name} ({s.city}, {s.country}) — {s.description}"
            for s in draft.stops
        )

        budget_context = (
            f"Total budget: ${request.budget_usd} USD"
            if request.budget_usd is not None
            else "No strict budget limit"
        )

        prompt = f"""You are an expert travel budget analyst.

Analyse the costs for the following trip and return a detailed budget breakdown in USD.

Trip details:
- Origin: {request.origin}
- Destination: {request.destination}
- Travelers: {request.travelers}
- Duration: {draft.days} days
- {budget_context}
- Interests: {", ".join(request.interests) if request.interests else "general travel"}

Itinerary stops:
{stops_summary}

Return a JSON object with these fields:
- "transportation": estimated total transportation cost in USD (flights, trains, local transport)
- "accommodation": estimated total accommodation cost in USD for all nights
- "food": estimated total food & dining cost in USD for all days
- "activities": estimated total activities & entrance fees in USD
- "total": sum of all above costs
- "per_person": total divided by number of travelers
- "notes": practical budget tips and assumptions in Traditional Chinese (繁體中文), about 100 characters

All monetary values must be numbers (not strings).
"""

        response = await self._openai.responses.create(
            model="gpt-4.1-mini",
            input=prompt,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "budget_analysis",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "transportation": {"type": "number", "minimum": 0},
                            "accommodation": {"type": "number", "minimum": 0},
                            "food": {"type": "number", "minimum": 0},
                            "activities": {"type": "number", "minimum": 0},
                            "total": {"type": "number", "minimum": 0},
                            "per_person": {"type": "number", "minimum": 0},
                            "notes": {"type": "string"},
                        },
                        "required": [
                            "transportation", "accommodation", "food",
                            "activities", "total", "per_person", "notes",
                        ],
                        "additionalProperties": False,
                    },
                    "strict": True,
                }
            },
        )

        data = json.loads(response.output_text)

        logger.info(
            "BudgetAgent: total=$%.0f, per_person=$%.0f for %s → %s (%d travelers)",
            data.get("total", 0),
            data.get("per_person", 0),
            request.origin,
            request.destination,
            request.travelers,
        )

        return BudgetAnalysis(
            transportation=data.get("transportation", 0.0),
            accommodation=data.get("accommodation", 0.0),
            food=data.get("food", 0.0),
            activities=data.get("activities", 0.0),
            total=data.get("total", 0.0),
            per_person=data.get("per_person", 0.0),
            notes=data.get("notes", ""),
        )
