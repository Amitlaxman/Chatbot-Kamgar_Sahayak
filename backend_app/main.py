import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pymongo import MongoClient
from fastapi.middleware.cors import CORSMiddleware
from langchain_groq import ChatGroq
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from sentence_transformers import CrossEncoder
import numpy as np
from typing import Optional, List, Dict

# Load environment variables
load_dotenv()

# --- Configuration ---
MONGO_URI = os.getenv("MONGO_URI")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DB_NAME = "legal_db"
VECTOR_COLLECTION_NAME = "legal_vectors"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Legal Chatbot Backend",
    description="A chatbot API using a RAG pipeline with MongoDB, Groq, and Re-ranking.",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Pydantic Models ---
class ChatRequest(BaseModel):
    query: str
    user_id: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    sources: List[Dict]


# --- Global Variables ---
db_client: Optional[MongoClient] = None
vector_store: Optional[MongoDBAtlasVectorSearch] = None
llm: Optional[ChatGroq] = None
reranker: Optional[CrossEncoder] = None
retriever = None


# --- Startup Event ---
@app.on_event("startup")
def startup_event():
    """Initialize all necessary models and database connections on server start."""
    global db_client, vector_store, llm, reranker, retriever

    print("🚀 Server starting up...")

    try:
        db_client = MongoClient(MONGO_URI)
        db_client.admin.command("ping")
        db = db_client[DB_NAME]
        collection = db[VECTOR_COLLECTION_NAME]

        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

        vector_store = MongoDBAtlasVectorSearch(
            collection=collection,
            embedding=embeddings,
            index_name="vector_index",  # <-- FIXED to use your index name
        )
        print("✅ MongoDB connection and vector store initialized.")
    except Exception as e:
        print(f"❌ Fatal: Could not connect to MongoDB. {e}")
        raise

    llm = ChatGroq(
        temperature=0,
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.1-8b-instant",
    )
    print("✅ Groq LLM initialized.")

    reranker = CrossEncoder(RERANKER_MODEL_NAME)
    print("✅ Re-ranker model loaded.")

    retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": 20}
    )
    print("✅ Retriever initialized.")
    print("--- Server startup complete. ---")


# --- Helper Functions ---
def generate_hypothetical_document(query: str, llm_instance) -> str:
    """Generates a hypothetical document from a query for improved retrieval."""
    hyde_prompt_template = """
    Please write a short, one-paragraph document that answers the following user question.
    This document should be written as if it were an excerpt from a legal guide for workers in Madhya Pradesh.
    QUESTION: {question}

    DOCUMENT:
    """
    hyde_prompt = PromptTemplate(
        template=hyde_prompt_template, input_variables=["question"]
    )
    hyde_chain = hyde_prompt | llm_instance | StrOutputParser()
    return hyde_chain.invoke({"question": query})


def rerank_documents(query: str, docs: list) -> list:
    """Re-ranks documents based on relevance to the original query."""
    if not docs:
        return []
    print(f"🔎 Reranking {len(docs)} documents...")
    pairs = [[query, doc.page_content] for doc in docs]
    scores = reranker.predict(pairs)
    doc_scores = list(zip(docs, scores))
    doc_scores.sort(key=lambda x: x[1], reverse=True)
    reranked_docs = [doc for doc, _ in doc_scores]
    print("✅ Re-ranking complete.")
    return reranked_docs


# --- Chat Endpoint ---
@app.post("/chat", response_model=ChatResponse)
async def chat_handler(request: ChatRequest):
    if not retriever or not llm:
        raise HTTPException(status_code=503, detail="Server not fully initialized.")

    query = request.query.strip()
    print(f"\n💬 Received query: '{query}'")

    greetings = ["hello", "hi", "hey", "greetings"]
    if query.lower() in greetings:
        return ChatResponse(
            response="Hello! How can I help you with Madhya Pradesh labor laws?",
            sources=[],
        )

    # 1a. Generate Hypothetical Document for embedding
    print("🧠 Generating hypothetical document for query (HyDE)...")
    hypothetical_doc = generate_hypothetical_document(query, llm)
    print(f"📝 Hypothetical Document Snippet: {hypothetical_doc[:150]}...")

    # 1b. Retrieve documents using the HyDE content
    print("🔍 Retrieving documents using HyDE...")
    initial_docs = retriever.invoke(hypothetical_doc)
    print(f"✅ Found {len(initial_docs)} initial documents.")

    if not initial_docs:
        return ChatResponse(
            response="I couldn't find any information related to your query.",
            sources=[],
        )

    # 2. Re-rank documents using the ORIGINAL query
    reranked_docs = rerank_documents(query, initial_docs)

    # 3. Select top documents for context
    top_k = 4  # Increased K slightly for broader context
    final_context_docs = reranked_docs[:top_k]
    context_text = "\n\n---\n\n".join(
        [doc.page_content for doc in final_context_docs]
    )

    # 4. Create an improved prompt for synthesis
    prompt_template = """
    You are a helpful assistant for labor laws in Madhya Pradesh, India.
    Your goal is to answer the user's question based on the context provided below.
    Do not say things like "Based on the provided context".
    Synthesize and summarize the information from the context to provide a clear and helpful answer.
    If the context contains relevant information, explain it in simple terms.
    If the context does not contain enough information, state that you couldn't find specific details in the available documents, say "I'll report this to an admin". Do not invent information.
    Cite your sources.
    If a source has a valid link (not "Not Available"), you can include it in your citation.

    CONTEXT:
    {context}

    QUESTION:
    {question}

    ANSWER:
    """
    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    # 5. Create and invoke the RAG chain
    rag_chain = (
        {"context": lambda _: context_text, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print("🤖 Invoking LLM chain...")
    answer = rag_chain.invoke(query)
    print(f"✅ Generated answer: {answer}")

    # 6. Format sources
    sources = [
        {
            "source_collection": doc.metadata.get("source_collection", "Unknown"),
            "content_snippet": doc.page_content[:150] + "...",
        }
        for doc in final_context_docs
    ]

    return ChatResponse(response=answer, sources=list({v["source_collection"]: v for v in sources}.values())) # Return unique sources


@app.get("/")
def read_root():
    return {"status": "ok", "message": "Legal Chatbot Backend is running."}

