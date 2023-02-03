from fastapi import APIRouter, HTTPException, status, Depends
from pymongo.database import Database

from typing import Dict

from src.database.init_db import get_db
from src.database.models.crud_item import (
    find_all,
    find_by_equip_category,
    find_by_item_id,
    find_by_item_idxs,
)

router = APIRouter()


@router.get("/items", description="모든 아이템 정보를 불러옵니다. ")
async def get_all_items(db: Database = Depends(get_db)) -> Dict:
    res = await find_all(db)
    if len(res) == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No item found",
        )
    return {"status": "success", "results": len(res), "items": res}


@router.get("/items/{category}", description="장비 카테고리에 따른 정보를 불러옵니다.")
async def get_items_by_category(category: str, db: Database = Depends(get_db)) -> Dict:
    res = await find_by_equip_category(category, db)
    if len(res) == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No item with equip category {category} found",
        )
    return {"status": "success", "results": len(res), "items": res}


@router.get("/item/{id}", description="id에 맞는 아이템을 가져옵니다.")
async def get_item_by_id(id: int, db: Database = Depends(get_db)) -> Dict:
    res = await find_by_item_id(id, db)
    if res is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No item with item id {id} found",
        )
    # return res
    return {"status": "success", "item": res}


@router.get("/items/idxs/{category}", description="장비 카테고리에 따른 정보를 불러옵니다.")
async def get_item_idxs_by_category(
    category: str, db: Database = Depends(get_db)
) -> Dict:
    res = await find_by_item_idxs(category, db)
    if len(res) == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No item with equip category {category} found",
        )
    return {"status": "success", "results": len(res), "items": res}
