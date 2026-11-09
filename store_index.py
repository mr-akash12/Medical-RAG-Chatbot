"""
Medical RAG Chatbot - Pinecone Index Management
------------------------------------------------
This script manages the ingestion and indexing of medical PDF documents
into the Pinecone vector database.

Main Responsibilities:
1. Load medical PDF documents from the data/ directory.
2. Split extracted text into smaller chunks.
3. Generate vector embeddings using a Hugging Face model.
4. Connect to the existing Pinecone index.
5. Upload or update document chunks and their metadata.

Purpose:
Keep the medical knowledge base updated without manually
re-running the Jupyter Notebook.

Note:
This script does not automatically delete outdated chunks
when source documents are modified or removed.
"""


import os
import hashlib
from pathlib import Path

from dotenv import load_dotenv
from pinecone import Pinecone

from src.helper import (
    load_pdf_file,
    text_split,
    download_hugging_face_embeddings
)


# 1. Load environment variables
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY is missing!")


def main():

    # 2. Load medical PDF documents
    extracted_data = load_pdf_file(
        data=str(BASE_DIR / "data")
    )

    # 3. Split documents into chunks
    text_chunks = text_split(extracted_data)

    if not text_chunks:
        raise ValueError("No text chunks found!")

    print(f"Total Chunks: {len(text_chunks)}")

    # 4. Load Hugging Face embeddings
    embeddings = download_hugging_face_embeddings()

    # 5. Connect to existing Pinecone index
    pc = Pinecone(api_key=PINECONE_API_KEY)

    index_name = "medicalbot"
    namespace = "medical-rag"

    if index_name not in [idx.name for idx in pc.indexes.list()]:
        raise ValueError(f"Index '{index_name}' not found!")

    index = pc.index(index_name)

    # 6. Generate embeddings and upload documents
    batch_size = 32
    total_uploaded = 0

    for start in range(0, len(text_chunks), batch_size):

        batch = text_chunks[start:start + batch_size]

        texts = [doc.page_content for doc in batch]
        vectors = embeddings.embed_documents(texts)

        documents = []

        for doc, vector in zip(batch, vectors):

            content_key = (
                str(doc.metadata.get("source", "")) +
                str(doc.metadata.get("page", "")) +
                doc.page_content
            )

            doc_id = hashlib.sha256(
                content_key.encode("utf-8")
            ).hexdigest()

            documents.append({
                "_id": doc_id,
                "vector": vector,
                "text": doc.page_content,
                "source": str(doc.metadata.get("source", "")),
                "page": int(doc.metadata.get("page", 0))
            })

        response = index.documents.upsert(
            namespace=namespace,
            documents=documents
        )

        total_uploaded += response.upserted_count

        print(
            f"Processed {total_uploaded}/{len(text_chunks)} chunks"
        )

    print("Pinecone documents updated successfully!")


if __name__ == "__main__":
    main()
