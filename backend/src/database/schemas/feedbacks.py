from pydantic import BaseModel


# 추천 결과에 대한 피드백 정보
class FeedbackItem(BaseModel):
    item_id: int
    item_index: int


class FeedbackSchema(BaseModel):
    Hat: FeedbackItem
    Hair: FeedbackItem
    Face: FeedbackItem
    Top: FeedbackItem
    Bottom: FeedbackItem
    Shoes: FeedbackItem
    Weapon: FeedbackItem
    is_positive: bool

    class Config:
        orm_mode = True
