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

app = FastAPI(
    title="AI Trip Planner",
    description="Trip planner using FastAPI, OpenAI, and a simple UI",
    version="1.0.0",
)


class TripRequest(BaseModel):
    destination: str = Field(..., example="Tokyo")
    budget_usd: Optional[int] = Field(
        None,
        example=1500,
        description="Optional budget. If empty, there is no limit.",
    )
    days: Optional[int] = Field(
        None,
        example=5,
        description="Optional trip length. If empty, the AI chooses a good duration.",
    )
    interests: List[str] = Field(
        default_factory=list,
        example=["food", "culture", "nature"],
    )


class TripResponse(BaseModel):
    destination: str
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
            f"The maximum budget is ${req.budget_usd} USD."
            if req.budget_usd is not None
            else "There is no budget limit."
        )

        days_text = (
            f"The trip should be {req.days} days."
            if req.days is not None
            else "The trip duration is flexible. Choose a practical duration."
        )

        interests_text = (
            ", ".join(req.interests)
            if req.interests
            else "general sightseeing, food, culture, and relaxation"
        )

        prompt = f"""
You are a helpful travel planner.

Create a clear and practical trip plan.

Destination: {req.destination}
Budget: {budget_text}
Duration: {days_text}
Interests: {interests_text}

Return the answer with:
1. Short trip summary
2. Suggested number of days
3. Day-by-day itinerary
4. Estimated cost breakdown
5. Food and transport suggestions
6. Important travel tips

Make the plan realistic and easy to follow.
"""

        response = client.responses.create(
            model="gpt-4.1-mini",
            input=prompt,
        )

        return TripResponse(
            destination=req.destination,
            budget_usd=req.budget_usd,
            days=req.days,
            plan=response.output_text,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))