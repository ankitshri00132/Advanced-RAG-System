from typing import List, Dict, Any
from fastembed.rerank.cross_encoder import TextCrossEncoder
from langsmith import traceable

# Initialize the reranker
reranker = TextCrossEncoder("jinaai/jina-reranker-v1-tiny-en")

@traceable(name="reranking_chunks")
def get_reranked_documents(
    query: str, 
    retrieved_results: List[Dict[str, Any]], 
    top_k: int = 3
) -> List[Dict[str, Any]]:
    """
    Reranks initial retrieved search results and fetches them in sorted order.
    
    Args:
        query: The search query.
        retrieved_results: List of search results from hybrid search containing 'payload' or 'metadata'.
        top_k : Number of top reranked documents to return.

        
    Returns:
        A list of dictionaries containing rank, score, and the document contents.
    """
    if not retrieved_results:
        return []
        
        
    # Extract the document text from the retrieved results
    documents = []
    for item in retrieved_results:
        payload = item.get("payload")
        doc_text = payload.get("text")

        if doc_text:
            documents.append(doc_text)
        
    # Get the new similarity scores from the reranker
    rerank_scores = reranker.rerank(query, documents)
    
    # Generate ranks with their original index
    ranked_pairs = list(enumerate(rerank_scores))
    ranked_pairs.sort(key=lambda x: x[1], reverse=True)
    
    # Fetch the original documents in ranked order and structure the output
    final_ranked_docs = []
    for rank_idx, (original_index, score) in enumerate(ranked_pairs, start=1):
        original_item = retrieved_results[original_index]
        payload = original_item.get("payload",{}) 
        
        final_ranked_docs.append({
            "rank": rank_idx,
            "rerank_score": float(score),
            "original_score": original_item.get("score"),
            "document": payload.get("text",""),
            "metadata": {k: v for k, v in payload.items() if k not in ["text"]}
        })
        
    return final_ranked_docs[:top_k]
