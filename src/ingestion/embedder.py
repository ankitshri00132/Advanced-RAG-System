from qdrant_client import QdrantClient, models
import os
import uuid
from dotenv import load_dotenv
load_dotenv()


def embed_chunks(chunks: list,document_id,filename):
    client = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"), timeout=300)

    dense_vector_name = 'dense'
    sparse_vector_name = 'sparse'
    dense_model_name = 'BAAI/bge-base-en-v1.5'
    sparse_model_name = "Qdrant/bm42-all-minilm-l6-v2-attentions"

    # create a dict which contains page content and its metadata
    metadata_with_chunks = [
        {
            'document_id':document_id,
            'file_name':filename,
            'text': chunk.page_content,
            'page': chunk.metadata.get("page"),
            'title':chunk.metadata.get('title')
        }
        for chunk in chunks
    ]

    # create a collection if does not exist
    if not client.collection_exists('main_vector_store'):
        client.create_collection(
            collection_name='main_vector_store',
            vectors_config={
                dense_vector_name: models.VectorParams(
                    size=client.get_embedding_size(dense_model_name),
                    distance=models.Distance.COSINE
                )
            },
            sparse_vectors_config={
                sparse_vector_name: models.SparseVectorParams()
            },

        )
        # Qdrant Cloud requires explicit payload indexes for filtered queries
        client.create_payload_index(
            collection_name='main_vector_store',
            field_name='document_id',
            field_schema=models.PayloadSchemaType.KEYWORD
        )
        print("Collection created !")
    # uploading the embeddings to the db
    client.upload_collection(
        collection_name='main_vector_store',
        vectors=[
            {
                dense_vector_name: models.Document(text=chunk.page_content, model=dense_model_name),
                sparse_vector_name: models.Document(text=chunk.page_content, model=sparse_model_name)
            }
            for chunk in chunks
        ],
        payload=metadata_with_chunks,
        ids=[str(uuid.uuid4()) for _ in range(len(chunks))]
    )
    print("Embeddings updated in Qdrant DB !")
    print("Embedding Dimension : ", client.get_embedding_size(dense_model_name))
