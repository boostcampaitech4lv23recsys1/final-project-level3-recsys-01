from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


host_name = "localhost"
port = 3306
user_name = "root"
password = ""


SQLALCHEMY_DATABASE_URL = (
    f"mysql+pymysql://{user_name}:{password}@{host_name}:{port}/items"
)
# SQLALCHEMY_DATABASE_URL = "postgresql://user:password@postgresserver/db"

engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
