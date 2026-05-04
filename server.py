import os
from typing import Optional, List

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI(title="AI Trip Planner")


# =========================
# Request Model
# =========================
class TripRequest(BaseModel):
    origin: str = Field(..., json_schema_extra={"example": "New York"})
    destination: str = Field(..., json_schema_extra={"example": "Tokyo"})
    travelers: int = Field(
        ...,
        description="Number of travelers",
        json_schema_extra={"example": 2}
    )

    # Optional fields
    budget_usd: Optional[int] = Field(
        None,
        description="Optional budget",
        json_schema_extra={"example": 2000}
    )

    days: Optional[int] = Field(
        None,
        description="Optional trip duration",
        json_schema_extra={"example": 5}
    )

    interests: List[str] = Field(
        default_factory=list,
        json_schema_extra={"example": ["food", "culture"]}
    )


# =========================
# Response Model
# =========================
class TripResponse(BaseModel):
    origin: str
    destination: str
    travelers: int
    budget_usd: Optional[int]
    days: Optional[int]
    plan: str


# =========================
# UI Route
# =========================
@app.get("/", include_in_schema=False)
def home():
    return FileResponse("ui.html")


@app.get("/health")
def health():
    return {"status": "ok"}


# =========================
# Main AI Endpoint
# =========================
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

Create a realistic and useful trip plan.

Starting location: {req.origin}
Destination: {req.destination}
Number of travelers: {req.travelers}
{budget_text}
{days_text}
Interests: {interests_text}

Requirements:
- Adjust costs based on number of travelers
- Include transportation from origin to destination
- Suggest suitable hotels based on group size
- Provide cost per person and total

Return:
1. Trip summary
2. Transportation plan
3. Estimated cost (per person + total)
4. Suggested duration
5. Day-by-day itinerary
6. Hotel suggestions
7. Food and transport tips
8. Important travel advice
"""

        response = client.responses.create(
            model="gpt-4.1-mini",
            input=prompt,
        )

        return TripResponse(
            origin=req.origin,
            destination=req.destination,
            travelers=req.travelers,
            budget_usd=req.budget_usd,
            days=req.days,
            plan=response.output_text,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))