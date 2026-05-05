import os
import json
from typing import Optional, List, Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from openai import OpenAI

# Optional MCP support
try:
    from fastapi_mcp import FastApiMCP
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY is missing. Add it to your .env file.")

client = OpenAI(api_key=api_key)

app = FastAPI(title="AI Trip Planner MCP Tools")


class TripRequest(BaseModel):
    origin: str = Field(..., description="Starting location")
    destination: str = Field(..., description="Trip destination")
    travelers: int = Field(..., description="Number of travelers")

    budget_usd: Optional[int] = None
    days: Optional[int] = None
    interests: List[str] = Field(default_factory=list)


class Stop(BaseModel):
    name: str
    day: int
    description: str
    lat: float
    lng: float


class TripResponse(BaseModel):
    origin: str
    destination: str
    travelers: int
    budget_usd: Optional[int]
    days: Optional[int]
    plan: str
    stops: List[Stop]


class StopsRequest(BaseModel):
    origin: str
    destination: str
    travelers: int
    budget_usd: Optional[int] = None
    days: Optional[int] = None
    interests: List[str] = Field(default_factory=list)


class StopsResponse(BaseModel):
    stops: List[Stop]


class RouteRequest(BaseModel):
    stops: List[Stop]


class RouteSegment(BaseModel):
    day: int
    color: str
    points: List[List[float]]


class RouteResponse(BaseModel):
    routes: List[RouteSegment]


@app.get("/", include_in_schema=False)
def home():
    return FileResponse("ui.html")


@app.get("/health", operation_id="health_check")
def health():
    return {"status": "ok"}


def get_day_color(day: int) -> str:
    colors = [
        "red", "blue", "green", "orange", "purple",
        "yellow", "violet", "grey", "black"
    ]
    return colors[(day - 1) % len(colors)]


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
- Put stops in correct travel order.
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
                                        "name": {"type": "string"},
                                        "day": {"type": "integer"},
                                        "description": {"type": "string"},
                                        "lat": {"type": "number"},
                                        "lng": {"type": "number"}
                                    },
                                    "required": ["name", "day", "description", "lat", "lng"],
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

        return TripResponse(
            origin=req.origin,
            destination=req.destination,
            travelers=req.travelers,
            budget_usd=req.budget_usd,
            days=req.days,
            plan=data["plan"],
            stops=data["stops"],
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/get-stops", response_model=StopsResponse, operation_id="get_stops")
def get_stops(req: StopsRequest):
    trip = plan_trip(
        TripRequest(
            origin=req.origin,
            destination=req.destination,
            travelers=req.travelers,
            budget_usd=req.budget_usd,
            days=req.days,
            interests=req.interests,
        )
    )

    return StopsResponse(stops=trip.stops)


@app.post("/get-route", response_model=RouteResponse, operation_id="get_route")
def get_route(req: RouteRequest):
    stops_by_day: dict[int, List[Stop]] = {}

    for stop in req.stops:
        stops_by_day.setdefault(stop.day, []).append(stop)

    routes = []

    for day in sorted(stops_by_day.keys()):
        day_stops = stops_by_day[day]

        points = [
            [stop.lat, stop.lng]
            for stop in day_stops
        ]

        if len(points) > 1:
            routes.append(
                RouteSegment(
                    day=day,
                    color=get_day_color(day),
                    points=points,
                )
            )

    return RouteResponse(routes=routes)


if MCP_AVAILABLE:
    mcp = FastApiMCP(app)
    mcp.mount()