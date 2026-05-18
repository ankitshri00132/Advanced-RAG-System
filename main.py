from fastapi import FastAPI
from src.api.router.ingest import router

app = FastAPI()

app.include_router(router)