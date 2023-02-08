from pymongo.database import Database
from motor.motor_asyncio import AsyncIOMotorDatabase

from typing import List, Dict

from src.database.schemas import ItemSchema

import json
from bson import json_util


async def find_all(db: AsyncIOMotorDatabase) -> List[ItemSchema]:
    res = await db.items.find({}).to_list(None)
    res = list(json.loads(json_util.dumps(res)))
    return res


async def find_by_equip_category(
    equip_category: str, db: AsyncIOMotorDatabase
) -> List[ItemSchema]:
    if equip_category == "Top":
        res = await db.items.find(
            {
                "$or": [
                    {"equip_category": equip_category},
                    {"equip_category": "Overall"},
                ]
            }
        ).to_list(None)
    else:
        res = await db.items.find({"equip_category": equip_category}).to_list(None)

    res = list(json.loads(json_util.dumps(res)))
    return res


async def find_by_item_id(item_id: int, db: AsyncIOMotorDatabase) -> Dict:
    res = await db.items.find_one({"item_id": item_id})

    res = json.loads(json_util.dumps(res))
    return res


async def find_by_item_idxs(equip_category: str, db: AsyncIOMotorDatabase) -> List[str]:
    if equip_category == "Top":
        res = db.items.find(
            {
                "$or": [
                    {"equip_category": equip_category},
                    {"equip_category": "Overall"},
                ],
                "is_cash": True,
            },
        )
    else:
        res = db.items.find(
            {"equip_category": equip_category, "is_cash": True},
        )
    res = await res.to_list(None)

    res = [d["index"] for d in (json.loads(json_util.dumps(res)))]

    return res


async def find_by_index(index: int, db: AsyncIOMotorDatabase) -> Dict:
    res = await db.items.find_one({"index": index})
    res = json.loads(json_util.dumps(res))
    return res
