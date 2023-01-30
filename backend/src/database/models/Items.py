from typing import List, Dict

from src.database.schemas import ItemSchema
from src.database.init_db import db


class Items:
    def find_all() -> List[ItemSchema]:
        res = list(db.items.find({}, {"_id": False}))
        return res

    def find_by_equip_category(equip_category: str) -> List[ItemSchema]:
        if equip_category == "Top":
            res = list(
                db.items.find(
                    {
                        "$or": [
                            {"equip_category": equip_category},
                            {"equip_category": "Overall"},
                        ]
                    },
                    {"_id": False},
                )
            )
        else:
            res = list(
                db.items.find({"equip_category": equip_category}, {"_id": False})
            )
        return res

    def find_by_item_id(item_id: int) -> Dict:
        res = db.items.find_one({"item_id": item_id}, {"_id": False})
        return res
