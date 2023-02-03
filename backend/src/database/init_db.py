from pymongo import MongoClient
import os
import certifi

ca = certifi.where()

# MongoDB URI
MONGODB_URI = os.getenv("MONGODB_URI")
# DB name
MONGODB_DATABASE_NAME = str(os.getenv("MONGODB_DATABASE_NAME"))

client = MongoClient(MONGODB_URI, tlsCAFile=ca)
if MONGODB_URI is None:
    print("env파일에서 URI를 읽어오지 못하고 있습니다.")
else:
    print("정상적으로 MongoDB 서버에 연결되었습니다.")
db = client[MONGODB_DATABASE_NAME]


async def get_db():
    try:
        yield db
    finally:
        pass
