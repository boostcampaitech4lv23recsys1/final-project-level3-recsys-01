from pydantic import BaseModel

from typing import Optional

from src.database.schemas import Item


# 유저가 집어 넣은 코디 정보
class CodiInfo(BaseModel):
    codi_hat: Optional[Item]
    codi_hair: Optional[Item]
    codi_face: Optional[Item]
    codi_top: Optional[Item]
    codi_bottom: Optional[Item]
    codi_shoes: Optional[Item]
    codi_weapon: Optional[Item]

    class Config:
        orm_mode = True
