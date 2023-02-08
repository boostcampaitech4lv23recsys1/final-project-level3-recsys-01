from pydantic import BaseModel

from typing import Optional


class InferenceInput(BaseModel):
    Hat: Optional[int]
    Hair: Optional[int]
    Face: Optional[int]
    Top: Optional[int]
    Bottom: Optional[int]
    Shoes: Optional[int]
    Weapon: Optional[int]


class ResultItem(BaseModel):
    item_id: int
    index: int
    name: str
    gcs_image_url: str


class InferenceResult(BaseModel):
    Hat: ResultItem
    Hair: ResultItem
    Face: ResultItem
    Top: ResultItem
    Bottom: ResultItem
    Shoes: ResultItem
    Weapon: ResultItem
