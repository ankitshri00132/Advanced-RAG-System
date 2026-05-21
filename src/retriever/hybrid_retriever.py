from dotenv import load_dotenv
load_dotenv()

from qdrant_client import QdrantClient,models

qdrant_client = QdrantClient(url='http://localhost:6333')

class HybridRetriever :
    DENSE_MODEL = 'BAAI/bge-small-en-v1.5'
    SPARSE_MODEL = 'Qdrant/bm42-all-minilm-l6-v2-attentions'
    dense_vector_name = 'dense'
    sparse_vector_name = 'sparse'

    def __init__(self,collection_name,qdrant_client):
        self.collection_name = collection_name
        self.qdrant_client = qdrant_client

    # search function

    def search(self,query_text : str):
        search_result = self.qdrant_client.query_points(
            collection_name = self.collection_name,
            prefetch = [
                models.Prefetch(
                    query = models.Document(text = query_text,model = self.DENSE_MODEL),
                    using = self.dense_vector_name,
                    limit = 10
                ),
                models.Prefetch(
                    query = models.Document(text = query_text,model= self.SPARSE_MODEL),
                    using = self.sparse_vector_name,
                    limit = 10
                ),
            ],
            query = models.FusionQuery(fusion=models.Fusion.RRF),
            limit = 10
        ).points

        result = []

        for point in search_result:
            result.append({
                "score":point.score,
                "payload":point.payload
            })
       
        return result