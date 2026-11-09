
"""
Medical RAG Chatbot - RAG Pipeline

Connects Pinecone vector search with a local Ollama LLM
using LangChain to generate context-based medical answers.
"""

import os

from dotenv import load_dotenv
from pinecone import Pinecone

from langchain_ollama import ChatOllama
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain
)

from src.prompt import prompt


# --------------------------------------------------
# Load Environment Variables
# --------------------------------------------------

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

load_dotenv(
    os.path.join(BASE_DIR, ".env")
)

PINECONE_API_KEY = os.getenv(
    "PINECONE_API_KEY"
)

INDEX_NAME = "medicalbot"
NAMESPACE = "medical-rag"


# --------------------------------------------------
# Create RAG Chain
# --------------------------------------------------

def create_rag_chain():

    if not PINECONE_API_KEY:
        raise ValueError(
            "PINECONE_API_KEY is missing! "
            "Please check your .env file."
        )

    # Initialize Pinecone
    pc = Pinecone(
        api_key=PINECONE_API_KEY
    )

    index = pc.index(
        INDEX_NAME
    )

    # Initialize Ollama LLM
    llm = ChatOllama(
        model="qwen3:4b-instruct",
        temperature=0.4,
        num_predict=800
    )

    # --------------------------------------------------
    # Retrieve Documents from Pinecone
    # --------------------------------------------------

    def retrieve_documents(inputs):

        question = inputs["input"]

        results = index.documents.search(
            namespace=NAMESPACE,

            score_by=[
                {
                    "type": "embed",
                    "field": "chunk_text",
                    "query": question
                }
            ],

            top_k=3,

            include_fields=[
                "chunk_text",
                "source",
                "page"
            ]
        )

        documents = []

        # Handle Pinecone search results
        hits = getattr(
            results,
            "matches",
            None
        )

        if hits is None:

            result_obj = getattr(
                results,
                "result",
                None
            )

            hits = getattr(
                result_obj,
                "hits",
                []
            )

        for hit in hits:

            # Extract document fields
            fields = getattr(
                hit,
                "fields",
                {}
            ) or {}

            # Extract text
            chunk_text = fields.get(
                "chunk_text",
                ""
            )

            if not chunk_text:
                continue

            # Extract score safely
            score = getattr(
                hit,
                "score",
                getattr(
                    hit,
                    "_score",
                    None
                )
            )

            # Create LangChain Document
            document = Document(
                page_content=chunk_text,

                metadata={
                    "source": fields.get(
                        "source"
                    ),

                    "page": fields.get(
                        "page"
                    ),

                    "score": score
                }
            )

            documents.append(
                document
            )

        return documents

    # --------------------------------------------------
    # Create Retriever
    # --------------------------------------------------

    retriever = RunnableLambda(
        retrieve_documents
    )

    # --------------------------------------------------
    # Create Question Answer Chain
    # --------------------------------------------------

    question_answer_chain = (
        create_stuff_documents_chain(
            llm,
            prompt
        )
    )

    # --------------------------------------------------
    # Create Final RAG Chain
    # --------------------------------------------------

    rag_chain = create_retrieval_chain(
        retriever,
        question_answer_chain
    )

    return rag_chain
