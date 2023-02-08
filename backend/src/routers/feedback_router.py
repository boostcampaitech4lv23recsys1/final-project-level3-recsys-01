from fastapi import APIRouter, HTTPException, status

from typing import Dict, Union

from src.database.schemas import FeedbackSchema
from src.utils import BigQueryHelper

router = APIRouter()
big_query_helper = BigQueryHelper(key_path="src/utils/gcs_key.json", dataset_name="log")


@router.post("/feedback", description="따봉 비따봉 결과 전송")
async def create_feedback(feedback: FeedbackSchema) -> Dict[str, Union[str, bool]]:
    feedback_dict = feedback.dict(exclude_none=False)

    # big query에 데이터 올리기
    big_query_data = {
        key.lower() + "_item_index": val["item_index"]
        for key, val in feedback_dict.items()
        if key != "is_positive"
    }
    big_query_data["is_positive"] = feedback_dict["is_positive"]
    big_query_return = big_query_helper.insert_dict_to_table("Feedback", big_query_data)
    if big_query_return is None:
        return {"status": "success", "feedback": big_query_data["is_positive"]}

    else:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Feedback sending to GCS Big Query has been failed",
        )
