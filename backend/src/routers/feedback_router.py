from fastapi import APIRouter, HTTPException, status
import json

from typing import Dict

from src.database.schemas import FeedbackSchema
from src.database.models.Feedbacks import Feedbacks

router = APIRouter()


@router.post("/feedback", description="모든 아이템 정보를 불러옵니다. ")
async def create_feedback(feedback: FeedbackSchema) -> Dict:
    feedback_dict = feedback.dict(exclude_none=False)
    res = Feedbacks.create(feedback_dict)
    return {"status": "success", "feedback": res}
