from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from sentence_transformers import SentenceTransformer
import os
import logging
from datetime import datetime
from typing import List, Dict, Any

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = os.getenv("DB_NAME", "chatbot_db")
FAQ_COLLECTION = os.getenv("FAQ_COLLECTION", "faqs")
LOGS_COLLECTION = os.getenv("LOGS_COLLECTION", "logs")
ADMIN_USERS_COLLECTION = os.getenv("ADMIN_USERS_COLLECTION", "admin_users")
NLP_MODEL_NAME = os.getenv("NLP_MODEL_NAME", "paraphrase-multilingual-MiniLM-L12-v2")

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Shramik Saathi Chatbot Backend",
    description="Backend API for the multilingual chatbot assisting laborers in MP, India.",
    version="0.1.0"
)

# --- Database Client ---
client: MongoClient = None
db = None

# --- NLP Model ---
nlp_model: SentenceTransformer = None

# --- FastAPI Event Handlers ---

@app.on_event("startup")
async def startup_db_client():
    """
    Connects to MongoDB and loads the NLP model on application startup.
    """
    global client, db, nlp_model
    try:
        logger.info("Connecting to MongoDB...")
        client = MongoClient(MONGO_URI)
        client.admin.command('ping') # Test connection
        db = client[DB_NAME]
        logger.info(f"Connected to MongoDB: {DB_NAME}")

        logger.info(f"Loading NLP model: {NLP_MODEL_NAME}...")
        nlp_model = SentenceTransformer(NLP_MODEL_NAME)
        logger.info("NLP model loaded successfully.")
    except ConnectionFailure as e:
        logger.error(f"MongoDB connection failed: {e}")
        # In a real app, you might want to exit or retry here
        raise HTTPException(status_code=500, detail="Could not connect to database.")
    except Exception as e:
        logger.error(f"Failed to load NLP model or other startup error: {e}")
        raise HTTPException(status_code=500, detail="Backend startup failed.")

@app.on_event("shutdown")
async def shutdown_db_client():
    """
    Closes the MongoDB connection on application shutdown.
    """
    if client:
        client.close()
        logger.info("MongoDB connection closed.")

# --- Pydantic Models for Request/Response ---

class ChatQuery(BaseModel):
    user_id: str
    query_text: str
    language: str = "en" # Default to English, can be 'hi', 'hinglish'

class ChatResponse(BaseModel):
    bot_response: str
    status: str # e.g., "answered", "unanswered"
    language: str
    query_id: str = None # Optional, if you generate a query ID

class LogEntry(BaseModel):
    timestamp: datetime
    user_id: str
    query_text: str
    bot_response_text: str
    status: str
    language: str
    similarity_score: float = None

# --- API Endpoints ---

@app.get("/")
async def read_root():
    """Basic root endpoint to confirm API is running."""
    return {"message": "Shramik Saathi Chatbot Backend is running!"}

@app.post("/chat", response_model=ChatResponse)
async def chat_with_bot(query: ChatQuery):
    """
    Processes a user's chat query, generates embeddings, and attempts to find an answer.
    """
    if not nlp_model:
        raise HTTPException(status_code=503, detail="NLP model not loaded. Please try again later.")
    if not db:
        raise HTTPException(status_code=503, detail="Database not connected. Please try again later.")

    user_query_text = query.query_text
    user_id = query.user_id
    language = query.language

    logger.info(f"Received chat query from {user_id} ({language}): '{user_query_text}'")

    # --- NLP: Generate Embedding for User Query ---
    try:
        user_embedding = nlp_model.encode(user_query_text, convert_to_tensor=False).tolist()
        logger.info(f"Generated embedding for query: {user_query_text[:30]}...")
    except Exception as e:
        logger.error(f"Error generating embedding for query '{user_query_text}': {e}")
        # Log this error and return a generic response
        await log_user_interaction(
            user_id=user_id,
            query_text=user_query_text,
            bot_response_text="Error processing your query. Please try again.",
            status="error",
            language=language
        )
        raise HTTPException(status_code=500, detail="Error processing your query.")

    # --- Placeholder for FAQ Matching Logic (Week 3 Task) ---
    # In Week 3, you will fetch FAQs from MongoDB, calculate similarity
    # between user_embedding and FAQ embeddings, and find the best match.
    # For now, we return a dummy response.

    bot_response_text = "I'm still learning! For now, I can only acknowledge your query."
    status = "unanswered" # Default to unanswered for now

    # --- Log User Interaction ---
    await log_user_interaction(
        user_id=user_id,
        query_text=user_query_text,
        bot_response_text=bot_response_text,
        status=status,
        language=language,
        # similarity_score=highest_similarity_score # Add this in Week 3
    )

    return ChatResponse(
        bot_response=bot_response_text,
        status=status,
        language=language
    )

@app.post("/log_query")
async def log_user_interaction(
    user_id: str,
    query_text: str,
    bot_response_text: str,
    status: str, # "answered", "unanswered", "error"
    language: str,
    similarity_score: float = None # Optional, for NLP-based logs
):
    """
    Logs user queries and bot responses to the database.
    This can be called internally by /chat or directly if needed.
    """
    if not db:
        logger.error("Database not connected for logging.")
        return {"message": "Logging failed: DB not connected."}

    log_entry = LogEntry(
        timestamp=datetime.now(),
        user_id=user_id,
        query_text=query_text,
        bot_response_text=bot_response_text,
        status=status,
        language=language,
        similarity_score=similarity_score
    )

    try:
        result = await db[LOGS_COLLECTION].insert_one(log_entry.dict())
        logger.info(f"Logged interaction with ID: {result.inserted_id}")
        return {"message": "Log entry created successfully", "id": str(result.inserted_id)}
    except Exception as e:
        logger.error(f"Failed to log interaction: {e}")
        raise HTTPException(status_code=500, detail="Failed to log interaction.")

# --- Admin Dashboard Endpoints (Basic - Week 3/4) ---
# These will be expanded with authentication and proper data retrieval later

@app.get("/admin/unanswered_queries")
async def get_unanswered_queries():
    """Retrieves all unanswered queries for admin review."""
    if not db:
        raise HTTPException(status_code=503, detail="Database not connected.")
    try:
        # In a real app, you'd add filtering, pagination, and authentication here
        queries = await db[LOGS_COLLECTION].find({"status": "unanswered"}).to_list(100) # Limit to 100 for example
        # Convert ObjectId to string for JSON serialization
        for q in queries:
            q["_id"] = str(q["_id"])
        return queries
    except Exception as e:
        logger.error(f"Failed to retrieve unanswered queries: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve data.")

@app.get("/admin/logs")
async def get_all_logs():
    """Retrieves all interaction logs."""
    if not db:
        raise HTTPException(status_code=503, detail="Database not connected.")
    try:
        # In a real app, you'd add filtering, pagination, and authentication here
        logs = await db[LOGS_COLLECTION].find({}).to_list(100) # Limit to 100 for example
        for l in logs:
            l["_id"] = str(l["_id"])
        return logs
    except Exception as e:
        logger.error(f"Failed to retrieve logs: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve data.")
    
