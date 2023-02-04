from fastapi import APIRouter, HTTPException, status, Depends
from pymongo.database import Database
from starlette.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, Dict, List

from collections import defaultdict
from src.AI import InferenceNewMF
from src.AI.inference import get_model

from src.database.init_db import get_db
from src.database.models.crud_item import (
    find_by_index,
)
from src.routers.schemas import InferenceInput, ResultItem, InferenceResult


router = APIRouter(prefix="/inference")


@router.post(
    "/submit/newMF",
    response_model=List[InferenceResult],
    description="codi recommendation",
)
async def newMF_output(
    equips: InferenceInput,
    model: InferenceNewMF = Depends(get_model),
    db: Database = Depends(get_db),
):
    """
    get_model_output 실행~
    :param equips:
    :return:
    """
    equips_dict = equips.dict(exclude_none=False)
    res = []

    predicts = await model.inference(equips_dict)
    parts = ["Hat", "Hair", "Face", "Top", "Bottom", "Shoes", "Weapon"]
    predicts = [list(map(lambda x: x[0], predict)) for predict in predicts]
    predicts = list(zip(*predicts))

    for idx in range(len(predicts)):
        codi_set = dict()
        for i, part in enumerate(parts):
            item = await find_by_index(predicts[idx][i], db)
            codi_set[part] = {
                "item_id": item["item_id"],
                "name": item["name"],
                "gcs_image_url": item["gcs_image_url"],
            }

        res.append(codi_set)
    return res


@router.get(
    "/submit/MCN",
    response_model=List[InferenceResult],
    description="codi recommendation",
)
async def mcn_output(equips: InferenceInput):
    """
    get_model_output 실행~
    :param equips:
    :return:
    """
    equips = dict(equips)
    predicts = MODELS["MCN"].inference(equips)

    return JSONResponse(predicts)
