from typing import List, Dict

from src.database.schemas import ItemSchema
from src.database.init_db import db

import json
from bson import json_util

class Items:
    def find_all() -> List[ItemSchema]:
        res = db.items.find({})
        res = list(json.loads(json_util.dumps(res)))
        return res

    def find_by_equip_category(equip_category: str) -> List[ItemSchema]:
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

    def find_by_item_id(item_id: int) -> Dict:
        res = db.items.find_one({"item_id": item_id})
        res = json.loads(json_util.dumps(res))

        return res

    def find_by_item_idxs(equip_category: str) -> List[str]:
        if equip_category == "Top":
            # res = db.items.find({
            #     "$and":
            #     [{
            #         "$or": [
            #             {"equip_category": equip_category},
            #             {"equip_category": "Overall"},
            #         ]},
            #         {"$not": {"overall_category": "dummy"}}
            #     ]
            # },
            #     {"index": 1, "_id": False},
            # )
            # res = [d["index"] for d in (json.loads(json_util.dumps(res)))]
            res = db.items.find({
                    "$and": [
                        {"equip_category": { "$in": [equip_category, "Overall"]}},
                        {"overall_category": {"$ne": "dummy"}}
                    ]
            },
                    {"index": 1, "_id": False}
            )
            res = [d["index"] for d in (json.loads(json_util.dumps(res)))]
        else:
            res = db.items.find({
                "$and": [
                    {"equip_category": equip_category},
                    {"overall_category": {"$ne": "dummy"}}
                ]},
                {"index": 1, "_id": False}
            )
            res = [d["index"] for d in (json.loads(json_util.dumps(res)))]

        return res