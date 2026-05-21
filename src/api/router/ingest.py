from fastapi import APIRouter, UploadFile, File
import shutil
from pathlib import Path
import uuid

from src.ingestion.loader import load_pdf
from src.ingestion.chunker import pdf_chunker
from src.ingestion.embedder import embed_chunks

router = APIRouter()

UPLOAD_DIR = Path("data/raw")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

@router.post("/ingest")
async def ingest_pdf(file: UploadFile = File(...)):

    file_path = UPLOAD_DIR / file.filename

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    print(f"{file.filename} uploaded successfully !")

    # loading the uploaded pdf
    documents = load_pdf(str(file_path))
    document_id = str(uuid.uuid4())
    page_count = len(documents)
    
    # chunking the document
    chunks = pdf_chunker(documents)
    chunks_count = len(chunks)

    # creating and storing the vectors of the chunks in qdrant db
    embed_chunks(chunks=chunks,document_id=document_id,filename = file.filename)

    return {
        "status":"success",
        "document_id":document_id,
        "pages_loaded":page_count,
        "chunks_created":chunks_count,
        "message": "Vectors successfully stored in Qdrant DB"
    }