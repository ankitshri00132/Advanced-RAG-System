from fastapi import FastAPI
from src.api.router.ingest import router as ingest_router
from src.api.router.retrieve import router as retrieve_router
from src.api.router.health import router as health_router

app = FastAPI()

app.include_router(ingest_router)
app.include_router(retrieve_router)
app.include_router(health_router)