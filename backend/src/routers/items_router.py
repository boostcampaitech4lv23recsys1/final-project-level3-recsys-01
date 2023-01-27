from fastapi import APIRouter, HTTPException, status

from typing import Dict

from src.database.models.Items import Items

router = APIRouter()


@router.get("/items", description="모든 아이템 정보를 불러옵니다. ")
async def get_all_items() -> Dict:
    res = Items.find_all()
    if len(res) == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No item found",
        )
    return {"status": "success", "results": len(res), "items": res}


@router.get("/items/{category}", description="장비 카테고리에 따른 정보를 불러옵니다.")
async def get_items_by_category(category: str) -> Dict:
    res = Items.find_by_equip_category(category)
    if len(res) == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No item with equip category {category} found",
        )
    return {"status": "success", "results": len(res), "items": res}


@router.get("/item/{id}", description="id에 맞는 아이템을 가져옵니다.")
async def get_item_by_id(id: int) -> Dict:
    res = Items.find_by_item_id(id)
    if res is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No item with item id {id} found",
        )
    # return res
    return {"status": "success", "item": res}
