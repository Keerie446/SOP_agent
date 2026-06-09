import os
import glob
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

DB_DIR = "./chroma_db"

def get_embeddings():
    # Use all-MiniLM-L6-v2, which is lightweight and good for general retrieval
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def ingest_docs(directory_path: str):
    """Loads documents from directory, chunks them, and stores in Chroma."""
    embeddings = get_embeddings()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    
    docs = []
    
    # Load PDFs
    for pdf_path in glob.glob(os.path.join(directory_path, "*.pdf")):
        print(f"Loading {pdf_path}...")
        loader = PyMuPDFLoader(pdf_path)
        docs.extend(loader.load())
        
    # Load MDs
    for md_path in glob.glob(os.path.join(directory_path, "*.md")):
        print(f"Loading {md_path}...")
        loader = TextLoader(md_path)
        docs.extend(loader.load())
        
    if not docs:
        print(f"No documents found in {directory_path}")
        return

    splits = text_splitter.split_documents(docs)
    
    print(f"Storing {len(splits)} chunks to Chroma...")
    db = Chroma.from_documents(splits, embeddings, persist_directory=DB_DIR)
    print("Ingestion complete.")

def get_retriever():
    """Returns a retriever for the Chroma vector store."""
    embeddings = get_embeddings()
    db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    return db.as_retriever(search_kwargs={"k": 4})

if __name__ == "__main__":
    # Example usage: python knowledge_base.py ./sops
    os.makedirs("./sops", exist_ok=True)
    print("Place dummy SOPs in ./sops and run again to ingest.")
    if os.listdir("./sops"):
         ingest_docs("./sops")
