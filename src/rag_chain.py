
"""
Medical RAG Chatbot - RAG Pipeline

Connects Pinecone vector search with a local Ollama LLM
using LangChain to generate context-based medical answers.
"""

import os

from dotenv import load_dotenv
from pinecone import Pinecone, DenseVectorQuery

from langchain_ollama import ChatOllama
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain
)

from src.helper import download_hugging_face_embeddings
from src.prompt import prompt


# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

INDEX_NAME = "medicalbot"
NAMESPACE = "medical-rag"


def create_rag_chain():

    if not PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY is missing!")

    # Initialize Pinecone
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.index(INDEX_NAME)

    # Initialize embedding model
    embeddings = download_hugging_face_embeddings()

    # Initialize Ollama
    llm = ChatOllama(
        model="qwen3:4b-instruct",
        temperature=0.4,
        num_predict=500
    )

    # Retrieve medical documents
    def retrieve_documents(inputs):

        question = inputs["input"]
        query_vector = embeddings.embed_query(question)

        results = index.documents.search(
            namespace=NAMESPACE,
            top_k=3,
            score_by=[
                DenseVectorQuery(
                    field="vector",
                    values=query_vector
                )
            ],
            include_fields=["text", "source", "page"]
        )

        documents = []

        for match in results.matches:
            documents.append(
                Document(
                    page_content=match.get("text") or "",
                    metadata={
                        "source": match.get("source"),
                        "page": match.get("page"),
                        "score": match.score
                    }
                )
            )

        return documents

    # Create Retriever
    retriever = RunnableLambda(retrieve_documents)

    # Create QA Chain
    question_answer_chain = create_stuff_documents_chain(
        llm,
        prompt
    )

    # Create RAG Chain
    rag_chain = create_retrieval_chain(
        retriever,
        question_answer_chain
    )

    return rag_chain
