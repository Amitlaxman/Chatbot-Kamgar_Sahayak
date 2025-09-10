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
    description="A bilingual chatbot API for Madhya Pradesh labor laws.",
    version="1.5.0", # Version updated for Hindi support
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

class Source(BaseModel):
    source_collection: str
    content_snippet: str
    source_link: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    sources: List[Source]


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
            index_name="vector_index",
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


# --- Language Processing Helper Functions ---

def detect_language(query: str, llm_instance: ChatGroq) -> str:
    """Detects if the query is primarily English or Hindi."""
    prompt = f"""
    Detect the primary language of the following text.
    Respond with only the two-letter language code (e.g., 'en' for English, 'hi' for Hindi/Hinglish).
    Text: "{query}"
    Language Code:
    """
    # MODIFIED: Access the .content attribute of the AIMessage object
    response = llm_instance.invoke(prompt).content.strip().lower()
    return "hi" if "hi" in response else "en"

def translate_text(text: str, target_lang: str, llm_instance: ChatGroq) -> str:
    """Translates text to the target language."""
    if target_lang == "en":
        prompt = f'Translate the following Hindi text to English:\n\nHindi: "{text}"\n\nEnglish:'
    else: # target_lang == "hi"
        prompt = f'Translate the following English text to Hindi:\n\nEnglish: "{text}"\n\nHindi:'
    # MODIFIED: Access the .content attribute of the AIMessage object
    return llm_instance.invoke(prompt).content.strip()


def generate_hypothetical_document(query: str, llm_instance) -> str:
    """Generates a hypothetical document from a query for improved retrieval."""
    hyde_prompt_template = """
    Please write a short, one-paragraph document that answers the following user question.
    This document should be written as if it were an excerpt from a legal guide for workers in Madhya Pradesh.
    QUESTION: {question}
    DOCUMENT:
    """
    hyde_prompt = PromptTemplate(template=hyde_prompt_template, input_variables=["question"])
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

    original_query = request.query.strip()
    print(f"\n💬 Received original query: '{original_query}'")

    # --- Bilingual Logic ---
    original_lang = detect_language(original_query, llm)
    print(f"🌍 Detected language: {original_lang}")

    if original_lang == 'hi':
        processing_query = translate_text(original_query, "en", llm)
        print(f"    -> Translated to English for processing: '{processing_query}'")
    else:
        processing_query = original_query
    # --- End Bilingual Logic ---

    print("🧠 Generating hypothetical document (HyDE)...")
    hypothetical_doc = generate_hypothetical_document(processing_query, llm)
    
    print("🔍 Retrieving documents using HyDE...")
    initial_docs = retriever.invoke(hypothetical_doc)
    print(f"✅ Found {len(initial_docs)} initial documents.")

    if not initial_docs:
        return ChatResponse(response="I couldn't find any information related to your query.", sources=[])

    reranked_docs = rerank_documents(processing_query, initial_docs)

    top_k = 4
    final_context_docs = reranked_docs[:top_k]
    
    context_parts = []
    for i, doc in enumerate(final_context_docs):
        source_name = doc.metadata.get("source_collection", "Unknown")
        source_link = doc.metadata.get("source_link", "Not Available")
        context_part = f"[Source {i+1}: {source_name} | Link: {source_link}]\n{doc.page_content}"
        context_parts.append(context_part)
    context_text = "\n\n---\n\n".join(context_parts)

    prompt_template = """
You are a helpful expert legal assistant for labour laws in Madhya Pradesh, India.

### Response Rules:
1. Always answer the user's question **based only on the provided CONTEXT**.
2. Write in a **clear, concise, and step-by-step** format (use bullet points or numbered lists when appropriate).
3. **Citations:**
   - After every fact, cite the source in this format: [Source X].
   - Do not invent or reorder source numbers. Only use the ones given in the CONTEXT.
4. **Sources Section:**
   - At the end of the answer, provide a "Sources" list.
   - Each source must include its number, the **collection name**, and its **link**.
   - Do not mention the same source more than once.
   - Example:
     Sources:
     1. Shops & Establishments Act – https://docs.google.com/xxxx
     2. Labour Notifications – https://drive.google.com/yyyy
5. If the answer is **not in the context**:
   - Say: "I cannot find this information in the provided documents."
   - Suggest a reasonable next step (e.g., consulting the Labour Department, visiting labour.mponline.gov.in).
   - End with: "I'll contact an admin."
6. Keep answers professional, accurate, and user-friendly.

### Additional Note:
- Always mention labour.mponline.gov.in if users need to apply, register, or get official forms online.

---
    CONTEXT:
    {context}
    QUESTION:
    {question}
    ANSWER:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    rag_chain = (
        {"context": lambda _: context_text, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print("🤖 Invoking LLM chain...")
    english_answer = rag_chain.invoke(processing_query)
    print(f"✅ Generated English answer: {english_answer}")

    # --- Final Translation Step ---
    if original_lang == 'hi':
        final_answer = translate_text(english_answer, "hi", llm)
        print(f"    -> Translated final answer back to Hindi.")
    else:
        final_answer = english_answer
    # --- End Final Translation ---

    sources = [
        Source(
            source_collection=doc.metadata.get("source_collection", "Unknown"),
            content_snippet=doc.page_content[:150] + "...",
            source_link=doc.metadata.get("source_link")
        )
        for doc in final_context_docs
    ]

    return ChatResponse(response=final_answer, sources=sources)


@app.get("/")
def read_root():
    return {"status": "ok", "message": "Legal Chatbot Backend is running."}

