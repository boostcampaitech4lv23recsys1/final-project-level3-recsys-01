from pymongo.database import Database

from typing import List, Dict

from src.database.schemas import ItemSchema

import json
from bson import json_util


async def find_all(db: Database) -> List[ItemSchema]:
    res = db.items.find({})
    res = list(json.loads(json_util.dumps(res)))
    return res


async def find_by_equip_category(equip_category: str, db: Database) -> List[ItemSchema]:
    if equip_category == "Top":
        res = db.items.find(
            {
                "$or": [
                    {"equip_category": equip_category},
                    {"equip_category": "Overall"},
                ]
            }
        )
        res = list(json.loads(json_util.dumps(res)))
    else:
        res = db.items.find({"equip_category": equip_category})
        res = list(json.loads(json_util.dumps(res)))

    return res


async def find_by_item_id(item_id: int, db: Database) -> Dict:
    res = db.items.find_one({"item_id": item_id})
    res = json.loads(json_util.dumps(res))

    return res


async def find_by_item_idxs(equip_category: str, db: Database) -> List[str]:
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
        res = [d["index"] for d in (json.loads(json_util.dumps(res)))]
    else:
        res = db.items.find(
            {"equip_category": equip_category, "is_cash": True},
        )
        res = [d["index"] for d in (json.loads(json_util.dumps(res)))]

    return res


async def find_by_index(index: int, db: Database) -> Dict:
    res = db.items.find_one({"index": index})
    res = json.loads(json_util.dumps(res))

    return res
