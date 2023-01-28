from fastapi import FastAPI
from uvicorn import run
from dotenv import load_dotenv
import os

from src.routers.items_router import router as items_router

load_dotenv(verbose=True)

# Port num
PORT = int(os.getenv("PORT"))

app = FastAPI()


@app.on_event("startup")
async def startup():
    print("startup")


app.include_router(items_router)

if __name__ == "__main__":
    run("src.main:app", host="127.0.0.1", port=PORT, reload=True)
