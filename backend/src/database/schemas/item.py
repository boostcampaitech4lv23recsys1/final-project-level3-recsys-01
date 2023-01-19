from pydantic import BaseModel


# 한 아이템이 가지는 정보
class Item(BaseModel):
    id: int
    requiredGender: int
    isCash: bool
    name: str
    category: str
    subCategory: str
    gcsImageUrl: str
    nameProcessed: str
    equipCategory: str

    class Config:
        orm_mode = True
