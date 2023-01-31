from pydantic import BaseModel


# 추천 결과에 대한 피드백 정보
class FeedbackSchema(BaseModel):
    hat_item_id: int
    hair_item_id: int
    face_item_id: int
    top_item_id: int
    bottom_item_id: int
    shoes_item_id: int
    weapon_item_id: int
    is_positive: bool

    class Config:
        orm_mode = True
