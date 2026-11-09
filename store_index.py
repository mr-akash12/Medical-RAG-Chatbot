
"""
Medical RAG Chatbot - Pinecone Index Management
------------------------------------------------
Loads medical PDFs, splits the text into chunks,
and uploads documents to Pinecone.
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

    # 2. Load PDF documents
    print("Loading PDF documents...")

    extracted_data = load_pdf_file(
        data=str(BASE_DIR / "data")
    )

    # 3. Split documents into chunks
    print("Splitting documents into chunks...")

    text_chunks = text_split(extracted_data)

    if not text_chunks:
        raise ValueError("No text chunks found!")

    print(f"Total Chunks: {len(text_chunks)}")

    # 4. Load embeddings model
    print("Loading Hugging Face embeddings...")

    embeddings = download_hugging_face_embeddings()

    # 5. Connect to Pinecone
    print("Connecting to Pinecone...")

    pc = Pinecone(api_key=PINECONE_API_KEY)

    index_name = "medicalbot"
    namespace = "medical-rag"

    available_indexes = [
        idx.name for idx in pc.indexes.list()
    ]

    if index_name not in available_indexes:
        raise ValueError(
            f"Index '{index_name}' not found!"
        )

    index = pc.index(index_name)

    # 6. Upload documents
    batch_size = 32
    total_uploaded = 0

    for start in range(
        0,
        len(text_chunks),
        batch_size
    ):

        batch = text_chunks[
            start:start + batch_size
        ]

        documents = []

        for doc in batch:

            content_key = (
                str(doc.metadata.get("source", ""))
                + str(doc.metadata.get("page", ""))
                + doc.page_content
            )

            doc_id = hashlib.sha256(
                content_key.encode("utf-8")
            ).hexdigest()

            # Pinecone schema requires chunk_text
            documents.append({
                "_id": doc_id,
                "chunk_text": doc.page_content,
                "source": str(
                    doc.metadata.get("source", "")
                ),
                "page": int(
                    doc.metadata.get("page", 0)
                )
            })

        response = index.documents.upsert(
            namespace=namespace,
            documents=documents
        )

        total_uploaded += response.upserted_count

        print(
            f"Uploaded {total_uploaded}/"
            f"{len(text_chunks)} chunks"
        )

    print(
        "Pinecone documents updated successfully!"
    )


if __name__ == "__main__":
    main()
