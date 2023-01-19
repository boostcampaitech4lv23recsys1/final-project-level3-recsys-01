from sqlalchemy.orm import Session

from src.database.table import ItemTable


def get_all_items(db: Session):
    return db.query(ItemTable).all()


def get_items_by_category(db: Session, category: str):
    return db.query(ItemTable).filter(ItemTable.equipCategory == category).all()


def get_item_by_id(db: Session, item_id: int):
    return db.query(ItemTable).filter(ItemTable.id == item_id).first()
