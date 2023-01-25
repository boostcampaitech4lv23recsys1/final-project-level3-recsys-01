from fastapi import FastAPI
from uvicorn import run
import os

from src.routers.all_router import router
from src.database.database_creation import create_db

# Port num
PORT = os.getenv("PORT")


def create_api():
    app = FastAPI()
    app.include_router(router)

    # 시작할 때 db 자동 생성
    @app.on_event("startup")
    async def startup():
        create_db()
        print("startup")

    return app


app = create_api()

if __name__ == "__main__":
    run("src.main:app", host="127.0.0.1", port=PORT, reload=True)
