from fastapi import APIRouter, HTTPException, status, Depends
from pymongo.database import Database
from starlette.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, Dict, List

from src.AI import MCNInference
from src.AI.init_model import get_model

from src.routers.schemas import InferenceInput, ResultItem, InferenceResult

router = APIRouter(prefix="/diagnosis")


@router.post(
    "/submit/MCN",
    response_model=float,
    description="codi recommendation",
)
async def MCN_diagnosis(
    equips: InferenceInput,
    model: MCNInference = Depends(get_model),
) -> float:
    """
    get_model_output 실행~
    :param equips:
    :return:
    """
    equips_dict = equips.dict(exclude_none=False)
    predict = await model.diagnosis(equips_dict)
    return predict
