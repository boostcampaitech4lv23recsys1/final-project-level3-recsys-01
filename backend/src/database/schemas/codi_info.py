from pydantic import BaseModel

from typing import Optional

from src.database.schemas import ItemSchema


# 유저가 집어 넣은 코디 정보
class CodiInfo(BaseModel):
    codi_hat: Optional[ItemSchema]
    codi_hair: Optional[ItemSchema]
    codi_face: Optional[ItemSchema]
    codi_top: Optional[ItemSchema]
    codi_bottom: Optional[ItemSchema]
    codi_shoes: Optional[ItemSchema]
    codi_weapon: Optional[ItemSchema]

    class Config:
        orm_mode = True
