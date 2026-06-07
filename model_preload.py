from fastembed import TextEmbedding, SparseTextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder

print("Downloading BGE...")
TextEmbedding(
    model_name="BAAI/bge-base-en-v1.5"
)

print("Downloading BM42...")
SparseTextEmbedding(
    model_name="Qdrant/bm42-all-minilm-l6-v2-attentions"
)

print("Downloading Jina Reranker...")
TextCrossEncoder(
    "jinaai/jina-reranker-v1-tiny-en"
)

print("All models downloaded.")