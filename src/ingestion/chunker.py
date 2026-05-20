from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter

def pdf_chunker(document : list) :
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 60
    )

    chunks = splitter.split_documents(document)

    return chunks