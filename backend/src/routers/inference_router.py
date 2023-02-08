from fastapi import APIRouter, HTTPException, status, Depends
from pymongo.database import Database
from starlette.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, Dict, List

from collections import defaultdict
from src.AI import InferenceNewMF, MCNInference
from src.AI.init_model import get_model

from src.database.init_db import get_db
from src.database.models.crud_item import (
    find_by_index,
)
from src.routers.schemas import InferenceInput, ResultItem, InferenceResult

import time


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


@router.post(
    "/submit/MCN",
    response_model=List[InferenceResult],
    description="mcn codi recommendation",
)
async def mcn_output(
    equips: InferenceInput,
    model: MCNInference = Depends(get_model),
    db: Database = Depends(get_db),
):
    equips_dict = equips.dict(exclude_none=False)
    res = list()

    predicts = await model.get_topk_codi(equips_dict)
    parts = ["Hat", "Hair", "Face", "Top", "Bottom", "Shoes", "Weapon"]

    for predict in predicts:
        codi_set = dict()
        overall_flag = False
        for part, item_idx in zip(parts, predict):
            if overall_flag and part == "Bottom":
                codi_set[part] = {
                    "item_id": -1,
                    "name": "상의가 한벌옷입니다",
                    "gcs_image_url": "https://storage.googleapis.com/maple_web/image/item/None.png",
                    "index": -1,
                }
                continue
            item_info = await find_by_index(item_idx, db)
            if item_info["equip_category"] == "Overall":
                overall_flag = True
            codi_set[part] = {
                "item_id": item_info["item_id"],
                "name": item_info["name"],
                "gcs_image_url": item_info["gcs_image_url"],
                "index": item_info["index"],
            }
        res.append(codi_set)

    return res
