import os
import json
from typing import Optional, List, Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from openai import OpenAI

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY is missing. Add it to your .env file.")

client = OpenAI(api_key=api_key)

app = FastAPI(title="AI Trip Planner")


class TripRequest(BaseModel):
    origin: str = Field(..., json_schema_extra={"example": "New York"})
    destination: str = Field(..., json_schema_extra={"example": "Tokyo"})
    travelers: int = Field(..., json_schema_extra={"example": 2})

    budget_usd: Optional[int] = Field(None, json_schema_extra={"example": 3000})
    days: Optional[int] = Field(None, json_schema_extra={"example": 5})
    interests: List[str] = Field(
        default_factory=list,
        json_schema_extra={"example": ["food", "anime", "culture"]}
    )


class TripResponse(BaseModel):
    origin: str
    destination: str
    travelers: int
    budget_usd: Optional[int]
    days: Optional[int]
    plan: str
    stops: List[dict[str, Any]]


@app.get("/", include_in_schema=False)
def home():
    return FileResponse("ui.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/plan-trip", response_model=TripResponse, operation_id="plan_trip")
def plan_trip(req: TripRequest):
    try:
        budget_text = (
            f"Budget is ${req.budget_usd} USD."
            if req.budget_usd is not None
            else "No strict budget limit."
        )

        days_text = (
            f"Trip duration is {req.days} days."
            if req.days is not None
            else "Trip duration is flexible."
        )

        interests_text = (
            ", ".join(req.interests)
            if req.interests
            else "general travel, food, culture, sightseeing"
        )

        prompt = f"""
You are an expert travel planner.

Create a realistic trip plan.

Starting location: {req.origin}
Destination: {req.destination}
Number of travelers: {req.travelers}
{budget_text}
{days_text}
Interests: {interests_text}

Important:
- Hotel cost MUST be included.
- Include cost per person and total cost.
- Adjust cost for number of travelers.
- Include transportation from origin to destination.
- Include all important trip stops with latitude and longitude.
- Include origin and destination in map stops.

The plan text must include:
1. Trip summary
2. Transportation plan
3. Estimated cost breakdown:
   - Transportation
   - Hotel / accommodation
   - Food
   - Local transport
   - Activities
   - Emergency / extra money
   - Total estimated cost
   - Estimated cost per person
4. Suggested duration
5. Day-by-day itinerary
6. Hotel recommendations
7. Food suggestions
8. Local transport suggestions
9. Important travel tips
"""

        response = client.responses.create(
            model="gpt-4.1-mini",
            input=prompt,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "trip_plan_response",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "plan": {
                                "type": "string"
                            },
                            "stops": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "name": {"type": "string"},
                                        "day": {"type": "integer"},
                                        "description": {"type": "string"},
                                        "lat": {"type": "number"},
                                        "lng": {"type": "number"}
                                    },
                                    "required": [
                                        "name",
                                        "day",
                                        "description",
                                        "lat",
                                        "lng"
                                    ],
                                    "additionalProperties": False
                                }
                            }
                        },
                        "required": ["plan", "stops"],
                        "additionalProperties": False
                    },
                    "strict": True
                }
            }
        )

        raw_text = response.output_text

        if not raw_text:
            raise HTTPException(
                status_code=500,
                detail="OpenAI returned an empty response."
            )

        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=500,
                detail=f"OpenAI did not return valid JSON: {raw_text}"
            )

        return TripResponse(
            origin=req.origin,
            destination=req.destination,
            travelers=req.travelers,
            budget_usd=req.budget_usd,
            days=req.days,
            plan=data["plan"],
            stops=data["stops"],
        )

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))