import os
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
    travelers: int = Field(..., json_schema_extra={"example": 2})

    budget_usd: Optional[int] = Field(None, json_schema_extra={"example": 2000})
    days: Optional[int] = Field(None, json_schema_extra={"example": 5})
    interests: List[str] = Field(
        default_factory=list,
        json_schema_extra={"example": ["food", "culture"]}
    )


class TripResponse(BaseModel):
    origin: str
    destination: str
    travelers: int
    budget_usd: Optional[int]
    days: Optional[int]
    plan: str


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
            else "Trip duration is flexible. Choose a practical duration."
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

IMPORTANT REQUIREMENTS:
- Include ALL costs clearly.
- Hotel cost MUST be included.
- Show cost per person AND total cost.
- Adjust all costs based on the number of travelers.
- Include transportation from origin to destination.
- Suggest suitable hotels based on the number of travelers.

Cost breakdown MUST include:
1. Transportation from origin to destination
2. Hotel / accommodation
3. Food
4. Local transport
5. Activities
6. Emergency / extra money

Use this exact cost format:

Estimated Cost Breakdown:
- Transportation: $X
- Hotel / Accommodation: $X total
  Explain as: $X per night × N nights × room count
- Food: $X
- Local Transport: $X
- Activities: $X
- Emergency / Extra: $X
- Total Estimated Cost: $X
- Estimated Cost Per Person: $X

Return:
1. Trip summary
2. Transportation plan
3. Estimated cost breakdown
4. Suggested duration
5. Day-by-day itinerary
6. Hotel recommendations
7. Food suggestions
8. Local transport suggestions
9. Important travel tips

Make the plan practical, realistic, and easy to understand.
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