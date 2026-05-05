import os
import json
from typing import Optional, List

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
    travelers: int = Field(..., ge=1, json_schema_extra={"example": 2})

    budget_usd: Optional[int] = Field(None, ge=0, json_schema_extra={"example": 3000})
    days: Optional[int] = Field(None, ge=1, le=30, json_schema_extra={"example": 5})
    interests: List[str] = Field(
        default_factory=list,
        json_schema_extra={"example": ["food", "anime", "culture"]}
    )


class Stop(BaseModel):
    """Enhanced Stop model with full Attraction_Card fields."""

    # Core fields (existing)
    name: str
    day: int
    description: str
    lat: float = Field(..., ge=-90.0, le=90.0)
    lng: float = Field(..., ge=-180.0, le=180.0)

    # New fields (task 2.1)
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
        description="每日交通建議清單，每項包含 day、mode、description、estimated_time"
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

Create a realistic and detailed trip plan.

Starting location: {req.origin}
Destination: {req.destination}
Number of travelers: {req.travelers}
{budget_text}
{days_text}
Interests: {interests_text}

Return JSON only.

The JSON must contain:
1. "plan": a detailed human-readable travel plan
2. "stops": map stop points with latitude and longitude

The "plan" text must include:

1. Trip Summary
2. Transportation From Origin To Destination
3. Estimated Cost Breakdown
- Transportation
- Hotel / accommodation
- Hotel cost per night
- Number of nights
- Room count assumption
- Food
- Local transport
- Activities
- Emergency / extra money
- Total estimated cost
- Estimated cost per person

4. Day-by-Day Itinerary
For each day include:
- Morning plan
- Afternoon plan
- Evening plan
- Main stops
- Estimated daily cost

5. Hotel Recommendations
6. Food Suggestions
7. Local Transport Suggestions
8. Important Travel Tips
9. Final Recommendation

Map stop rules:
- Include origin and destination.
- Include major stops from each itinerary day.
- Each stop must have: name, day, description, lat, lng.
- Each stop day must match the itinerary day.
- Put stops in the correct travel order so the map route line is useful.
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
                            "plan": {"type": "string"},
                            "stops": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "name":         {"type": "string"},
                                        "day":          {"type": "integer", "minimum": 1},
                                        "time":         {"type": "string"},
                                        "description":  {"type": "string"},
                                        "lat":          {"type": "number", "minimum": -90.0, "maximum": 90.0},
                                        "lng":          {"type": "number", "minimum": -180.0, "maximum": 180.0},
                                        "rating":       {"type": "number", "minimum": 0.0, "maximum": 5.0},
                                        "review_count": {"type": "integer", "minimum": 0},
                                        "category":     {"type": "string"},
                                        "ai_notes":     {"type": "string"},
                                        "website_url":  {"type": ["string", "null"]},
                                        "image_urls":   {
                                            "type": "array",
                                            "items": {"type": "string"},
                                            "maxItems": 5
                                        }
                                    },
                                    "required": [
                                        "name", "day", "time", "description",
                                        "lat", "lng", "rating", "review_count",
                                        "category", "ai_notes", "website_url", "image_urls"
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

        data = json.loads(response.output_text)

        stops = [Stop(**s) for s in data["stops"]]

        return TripResponse(
            origin=req.origin,
            destination=req.destination,
            travelers=req.travelers,
            budget_usd=req.budget_usd,
            days=req.days,
            plan=data["plan"],
            stops=stops,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))