from fastapi import APIRouter, HTTPException, status
from starlette.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List

from collections import defaultdict
from src.AI import InferenceNewMF
from src.AI.config import MODEL_CONFIG


router = APIRouter(prefix="/inference")

MODELS = {
    "newMF": InferenceNewMF(
        model_config=MODEL_CONFIG["newMF"],
    ),
}

class InputItem(BaseModel):
    Hat: Optional[int]
    Hair: Optional[int]
    Face: Optional[int]
    Top: Optional[int]
    Bottom: Optional[int]
    Shoes: Optional[int]
    Weapon: Optional[int]


class OutputItem(BaseModel):
    Hat: List[int]
    Hair: List[int]
    Face: List[int]
    Top: List[int]
    Bottom: List[int]
    Shoes: List[int]
    Weapon: List[int]


@router.post("/submit/newMF", response_model=OutputItem, description="codi recommendation")
async def newMF_output(equips: InputItem):
    """
    get_model_output 실행~
    :param equips:
    :return:
    """
    response = defaultdict(list)
    equips = dict(equips)

    predicts = MODELS["newMF"].inference(equips)
    predicts = [list(map(lambda x: x[0], predict)) for predict in predicts]

    for part, predicts in zip(["Hat", "Hair", "Face", "Top", "Bottom", "Shoes", "Weapon"], predicts):
        response[part] = predicts

    return response


@router.get("/submit/MCN", description="codi recommendation")
async def mcn_output(equips: InputItem):
    """
    get_model_output 실행~
    :param equips:
    :return:
    """
    equips = dict(equips)
    predicts = MODELS["MCN"].inference(equips)

    return JSONResponse(predicts)
