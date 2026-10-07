"""
ingest.py
Build the ChromaDB vector store from dip_knowledge.txt.
Run: python ingest.py
"""

import os
import shutil
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

KNOWLEDGE_FILE = "dip_knowledge.txt"
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "dip_diet"


def build_vector_store():
    print("[ingest] Loading knowledge file:", KNOWLEDGE_FILE)
    loader = TextLoader(KNOWLEDGE_FILE, encoding="utf-8")
    documents = loader.load()

    print("[ingest] Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    chunks = splitter.split_documents(documents)
    print(f"[ingest] Created {len(chunks)} chunks.")

    # Wipe old DB so we start fresh
    if os.path.exists(CHROMA_DB_PATH):
        shutil.rmtree(CHROMA_DB_PATH)

    print("[ingest] Building ChromaDB vector store at:", CHROMA_DB_PATH)
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    ef = DefaultEmbeddingFunction()
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
    )

    texts = [c.page_content for c in chunks]
    ids   = [str(i) for i in range(len(chunks))]
    collection.add(documents=texts, ids=ids)

    print(f"[ingest] Vector store built with {collection.count()} chunks. Done!")
    return collection


if __name__ == "__main__":
    build_vector_store()
