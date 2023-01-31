from fastapi import APIRouter, HTTPException, status
from starlette.responses import JSONResponse
from pydantic import BaseModel

from src.AI import InferenceNewMF
from src.AI.config import MODEL_CONFIG


router = APIRouter(prefix="/inference")

MODELS = {
    "newMF": InferenceNewMF(
        model_config=MODEL_CONFIG["newMF"],
    ),
}

class TransferItem(BaseModel):
    Hat: int
    Hair: int
    Face: int
    Top: int
    Bottom: int
    Shoes: int
    Weapon: int


@router.get("/submit/newMF", description="codi recommendation")
async def newMF_output(equips: TransferItem):
    """
    get_model_output 실행~
    :param equips:
    :return:
    """
    # equips = dict(equips)
    # predicts = MODELS["newMF"].inference(equips)

    # return JSONResponse(predicts)
    return {"status": "success", "item": equips}


@router.get("/submit/MCN", description="codi recommendation")
async def mcn_output(equips: TransferItem):
    """
    get_model_output 실행~
    :param equips:
    :return:
    """
    equips = dict(equips)
    predicts = MODELS["MCN"].inference(equips)

    return JSONResponse(predicts)
