from pymongo import MongoClient
import os

# MongoDB URI
MONGODB_URI = os.getenv("MONGODB_URI")
# DB name
MONGODB_DATABASE_NAME = str(os.getenv("MONGODB_DATABASE_NAME"))

client = MongoClient(MONGODB_URI)
print("정상적으로 MongoDB 서버에 연결되었습니다.")

db = client[MONGODB_DATABASE_NAME]
