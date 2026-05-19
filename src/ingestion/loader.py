from langchain_community.document_loaders import PyMuPDFLoader

def load_pdf(file_path : str):

    file = PyMuPDFLoader(file_path)
    documents = file.load()
    return documents


