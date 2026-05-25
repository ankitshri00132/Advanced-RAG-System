from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.graph.rag_graph import build_graph

router = APIRouter()

rag_app = build_graph()


class SearchRequest(BaseModel):
    query: str


@router.post('/retrieve')
async def retrieve_chunks(request: SearchRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:

        result = rag_app.invoke({         # --> result has the final_state of the graph
            'query': request.query
        })

        return {
            'query': request.query,
            'answer': result['answer'],
            'sources': result['reranked_results']
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
