from qdrant_client import QdrantClient, models
from dotenv import load_dotenv
load_dotenv()


def embed_chunks(chunks: list):
    client = QdrantClient(url='http://localhost:6333')

    dense_embedding_model = "BAAI/bge-small-en-v1.5"

    # create a dict which contains page content and its metadata
    metadata_with_chunks = [
        {
            'document': chunk.page_content,
            'source': chunk.metadata
        }
        for chunk in chunks
    ]
    
    # create a collection if does not exist
    if not client.collection_exists('main_vector_store'):
        client.create_collection(
            collection_name='main_vector_store',
            vectors_config=models.VectorParams(
                size=client.get_embedding_size(dense_embedding_model_name),
                distance=models.Distance.COSINE
            ),
        )

    else:
        client.upload_collection(
            collection_name='main_vector_store',
            vectors=[models.Document(
                text=chunk.page_content, model=dense_embedding_model_name) for chunk in chunks],
            payload=metadata_with_chunks,
            ids=list(range(len(chunks))),
        )
