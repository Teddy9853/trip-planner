"""
Transport Agent — provides daily transport suggestions via GPT-4.1-mini.

For each day in the itinerary it recommends transport modes (public transit,
taxi, walking) with estimated travel times between stops.

Requirement: 7.5
"""

from __future__ import annotations

import json
import logging
import os
from typing import TYPE_CHECKING

from openai import AsyncOpenAI

if TYPE_CHECKING:
    from agents.itinerary import ItineraryDraft
    from server import TransportSuggestions

logger = logging.getLogger(__name__)


class TransportAgent:
    """
    Generates daily transport suggestions using GPT-4.1-mini.

    Usage::

        agent = TransportAgent()
        suggestions = await agent.suggest(draft)
    """

    def __init__(self) -> None:
        self._openai = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    async def suggest(self, draft: "ItineraryDraft") -> "TransportSuggestions":
        """
        Produce TransportSuggestions for the given itinerary draft.

        Returns per-day transport recommendations plus general tips.

        Requirement: 7.5
        """
        # Import here to avoid circular imports at module load time
        from server import TransportSuggestions

        # Build a day-by-day stop summary for the prompt
        days_map: dict[int, list] = {}
        for stop in draft.stops:
            days_map.setdefault(stop.day, []).append(stop)

        days_text_parts = []
        for day_num in sorted(days_map.keys()):
            stops_on_day = days_map[day_num]
            stop_names = " → ".join(s.name for s in stops_on_day)
            days_text_parts.append(f"  Day {day_num}: {stop_names}")
        days_text = "\n".join(days_text_parts)

        prompt = f"""You are an expert travel logistics planner.

Provide practical transport suggestions for each day of the following itinerary.

Destination: {draft.destination}
Duration: {draft.days} days

Daily stops:
{days_text}

Return a JSON object with:
- "daily_suggestions": array of objects, one per day, each with:
  - "day": day number (integer)
  - "mode": primary transport mode (e.g. "地鐵 + 步行", "計程車", "公車", "租車")
  - "description": detailed transport instructions in Traditional Chinese (繁體中文)
  - "estimated_time": total estimated travel time between stops (e.g. "約 2 小時")
- "general_tips": overall transport tips for the destination in Traditional Chinese (繁體中文), about 100 characters

Cover all {draft.days} days.
"""

        response = await self._openai.responses.create(
            model="gpt-4.1-mini",
            input=prompt,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "transport_suggestions",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "daily_suggestions": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "day": {"type": "integer", "minimum": 1},
                                        "mode": {"type": "string"},
                                        "description": {"type": "string"},
                                        "estimated_time": {"type": "string"},
                                    },
                                    "required": ["day", "mode", "description", "estimated_time"],
                                    "additionalProperties": False,
                                },
                            },
                            "general_tips": {"type": "string"},
                        },
                        "required": ["daily_suggestions", "general_tips"],
                        "additionalProperties": False,
                    },
                    "strict": True,
                }
            },
        )

        data = json.loads(response.output_text)

        logger.info(
            "TransportAgent: generated suggestions for %d days in %s",
            len(data.get("daily_suggestions", [])),
            draft.destination,
        )

        return TransportSuggestions(
            daily_suggestions=data.get("daily_suggestions", []),
            general_tips=data.get("general_tips", ""),
        )
