from typing import List, Dict

from src.database.schemas import FeedbackSchema
from src.database.init_db import db

import json
from bson import json_util


class Feedbacks:
    def create(feedback) -> Dict:
        result_id = db.feedbacks.insert_one(feedback).inserted_id
        res = db.feedbacks.find_one({"_id": result_id})
        res = json.loads(json_util.dumps(res))
        return res
