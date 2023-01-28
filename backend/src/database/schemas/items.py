from pydantic import BaseModel


# 한 아이템이 가지는 정보
class ItemSchema(BaseModel):
    item_id: int
    required_gender: int
    is_cash: bool
    name: str
    category: str
    sub_category: str
    gcs_image_url: str
    name_processed: str
    equip_category: str

    class Config:
        orm_mode = True
