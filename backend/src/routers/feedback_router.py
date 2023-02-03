from fastapi import APIRouter, HTTPException, status
import json

from typing import Dict

from src.database.schemas import FeedbackSchema
from src.database.models.Feedbacks import Feedbacks
from src.utils import BigQueryHelper

router = APIRouter()
big_query_helper = BigQueryHelper(key_path="src/utils/gcs_key.json", dataset_name="log")


@router.post("/feedback", description="따봉 비따봉 결과 전송")
async def create_feedback(feedback: FeedbackSchema) -> Dict:
    feedback_dict = feedback.dict(exclude_none=False)
    res = Feedbacks.create(feedback_dict)
    # big_query_helper.insert_dict_to_table(res)
    return {"status": "success", "feedback": res}
