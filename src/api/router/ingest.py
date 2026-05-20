from fastapi import APIRouter, UploadFile, File
import shutil
from pathlib import Path

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
    
    # chunking the document
    chunks = pdf_chunker(documents)

    # creating and storing the vectors of the chunks in qdrant db
    embed_chunks(chunks)

    return {
        "message": "Vectors successfully stored in Qdrant DB"
    }