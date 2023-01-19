from fastapi import APIRouter, Depends

from sqlalchemy.orm import Session

from typing import List, Optional

from src.database.schemas import Item
from src.database.models import crud


from src.database.init_db import get_db

router = APIRouter(prefix="/item")


@router.get("/", description="모든 아이템 정보를 불러옵니다. ")
def get_all_items(db: Session = Depends(get_db)) -> List[Item]:
    items = crud.get_all_items(db)
    return items


@router.get("/{category}", description="모든 모자 정보를 불러옵니다.")
def get_items_by_category(category: str, db: Session = Depends(get_db)) -> List[Item]:
    items = crud.get_items_by_category(db, category=category)
    return items


# 이거 안되는 이유 알았음
# 위에서 /item/category를 이미 category로 아이템 가져오기로 할당했는데
# 여기서 /item/1 이걸로 아이템 가져오려고 해서 같은 방식이라 오류가 나는 것 같다.
# 혼자 해결할 수는 있는데, 일단 남겨두고 구조 정리해서 깃에 올리자.
@router.get("/{id}", description="id에 맞는 아이템을 가져옵니다.")
def get_item_by_id(id: int, db: Session = Depends(get_db)) -> Optional[Item]:
    item = crud.get_item_by_id(db, id=id)
    return item
