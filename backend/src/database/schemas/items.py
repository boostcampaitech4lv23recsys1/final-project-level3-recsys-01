from pydantic import BaseModel

# 한 아이템이 가지는 정보
class ItemSchema(BaseModel):
    index: int
    required_jobs: str
    required_level: int
    required_gender: int
    is_cash: bool
    desc: str
    item_id: int
    name: str
    overall_category: str
    category: str
    sub_category: str
    low_item_id: int
    high_item_id: int
    image_url: str
    gcs_image_url: str
    name_processed: str
    equip_category: str
    local_image_path: str

    class Config:
        orm_mode = True
