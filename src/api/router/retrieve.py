from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.graph.rag_graph import build_graph as rag
from src.graph.crag_graph import build_graph as crag

router = APIRouter()

rag_app = rag()
crag_app = crag()


class SearchRequest(BaseModel):
    query: str
    document_id: str = None


@router.post('/retrieve')
async def retrieve_chunks(request: SearchRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:

        result = await crag_app.ainvoke({         # --> result has the final_state of the graph
            'query': request.query,
            'document_id': request.document_id
        })

        return {
            'query': request.query,
            'answer': result['answer'],
            'sources': result.get('reranked_results', [])
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
