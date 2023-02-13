from motor.motor_asyncio import AsyncIOMotorClient

import os
import certifi

ca = certifi.where()

# MongoDB URI
MONGODB_URI = os.getenv("MONGODB_URI")
# DB name
MONGODB_DATABASE_NAME = str(os.getenv("MONGODB_DATABASE_NAME"))


async def get_db():
    client = AsyncIOMotorClient(MONGODB_URI, tlsCAFile=ca)
    db = client[MONGODB_DATABASE_NAME]
    return db
