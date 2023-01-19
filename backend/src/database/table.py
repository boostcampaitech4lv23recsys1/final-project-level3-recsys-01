from sqlalchemy import Boolean, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base


Base = declarative_base()


class ItemTable(Base):
    __tablename__ = "backend"

    id = Column(Integer, primary_key=True, index=True)
    requiredGender = Column(String)
    isCash = Column(Boolean)
    name = Column(String)
    category = Column(String)
    subCategory = Column(String)
    gcsImageUrl = Column(String)
    nameProcessed = Column(String)
    equipCategory = Column(String)
