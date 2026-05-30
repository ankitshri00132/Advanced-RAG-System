from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter

def pdf_chunker(document : list) :
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1500,
        chunk_overlap = 300
    )

    chunks = splitter.split_documents(document)

    return chunks