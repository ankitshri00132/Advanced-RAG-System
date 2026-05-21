from fastapi import APIRouter
from pydantic import BaseModel

from src.retriever.hybrid_retriever import HybridRetriever

router = APIRouter()

class SearchRequest(BaseModel):
    query : str

@router.post('/retrieve')
async def retrieve_chunks(query : SearchRequest):
    searcher = HybridRetriever(collection_name= "main_vector_store")

    result = searcher.search(text= query.query)

    return {
        "query":query.query,
        "result":result
        }
