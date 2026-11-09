
"""
Medical RAG Chatbot - Flask Application

Provides the web interface for the medical chatbot.
Uses LangChain, Pinecone, and a local Ollama LLM
to answer questions based on retrieved medical documents.
"""

from src.rag_chain import create_rag_chain
from flask import Flask, render_template, request, jsonify
import os


# Initialize Flask application
app = Flask(__name__)

# Initialize RAG pipeline once at startup
rag_chain = create_rag_chain()


# Home Page
@app.route("/")
def index():
    return render_template("chat.html")


# Chat Endpoint

@app.route("/get", methods=["POST"])
def chat():

    msg = request.form.get("msg", "").strip()

    if not msg:
        return jsonify({
            "error": "Please enter a question."
        }), 400

    response = rag_chain.invoke({"input": msg})

    answer = response["answer"]

    # Extract retrieved document sources
    sources = []

    for doc in response.get("context", []):
        sources.append({
            "file": os.path.basename(
                str(doc.metadata.get("source", "Unknown"))
            ),
            "page": doc.metadata.get("page")
        })

    # Remove duplicate sources
    unique_sources = list({
        (src["file"], src["page"]): src
        for src in sources
    }.values())
    print("Retrieved Sources:", unique_sources)

    return jsonify({
        "answer": answer,
        "sources": unique_sources
    })



# Run Flask Application
if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=8080,
        debug=False
    )
