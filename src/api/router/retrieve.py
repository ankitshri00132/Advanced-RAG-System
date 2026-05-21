from fastapi import APIRouter,HTTPException
from pydantic import BaseModel

from src.retriever.hybrid_retriever import HybridRetriever,qdrant_client
from src.retriever.reranker import get_reranked_documents,reranker

router = APIRouter()

searcher = HybridRetriever(collection_name= "main_vector_store",qdrant_client=qdrant_client)

class SearchRequest(BaseModel):
    query : str
    top_k : int = 3

@router.post('/retrieve')
async def retrieve_chunks(request : SearchRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        retrieved_results = searcher.search(
            query_text= request.query
            )

        reranked_results = get_reranked_documents(
            query = request.query,
            retrieved_results= retrieved_results,
            top_k= request.top_k
            )

        return {
            "query":request.query,
            "result":reranked_results
            }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

