from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter

def pdf_chunker(document : list) :
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 200
    )

    chunks = splitter.split_documents(document)

    return chunks