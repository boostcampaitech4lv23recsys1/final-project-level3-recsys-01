from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from uvicorn import run
from dotenv import load_dotenv

load_dotenv(verbose=True)

from pymongo import MongoClient
import os
import certifi

from src.routers.items_router import router as items_router
from src.routers.feedback_router import router as feedback_router
from src.routers.inference_router import router as inference_router
from src.routers.diagnosis_router import router as diagnosis_router

# Port num
PORT = int(os.getenv("BACKEND_PORT"))

app = FastAPI()


@app.on_event("startup")
async def startup():
    print("startup")


app.include_router(items_router)
app.include_router(feedback_router)
app.include_router(inference_router)
app.include_router(diagnosis_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == "__main__":
    run("src.main:app", host="0.0.0.0", port=PORT, reload=True)
