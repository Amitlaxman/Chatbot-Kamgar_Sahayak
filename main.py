from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from sentence_transformers import SentenceTransformer
import os
import logging
from datetime import datetime
from typing import List, Dict, Any
from dotenv import load_dotenv


load_dotenv()


MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = os.getenv("DB_NAME", "chatbot_db")
FAQ_COLLECTION = os.getenv("FAQ_COLLECTION", "faqs")
LOGS_COLLECTION = os.getenv("LOGS_COLLECTION", "logs")
ADMIN_USERS_COLLECTION = os.getenv("ADMIN_USERS_COLLECTION", "admin_users")
NLP_MODEL_NAME = os.getenv("NLP_MODEL_NAME", "paraphrase-multilingual-MiniLM-L12-v2") # Good for multilingual

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

# --- Startup and Shutdown Events ---
@app.on_event("startup")
async def startup_db_client():
    """
    Connects to MongoDB and loads the NLP model on application startup.
    """
    global client, db, nlp_model
    try:
        logger.info("Connecting to MongoDB...")
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        client.admin.command('ping') # Test connection
        db = client[DB_NAME]
        logger.info(f"Connected to MongoDB: {DB_NAME}")
        logger.info(f"Loading NLP model: {NLP_MODEL_NAME}...")
        nlp_model = SentenceTransformer(NLP_MODEL_NAME)
        logger.info("NLP model loaded successfully.")

    except ConnectionFailure as e:
        logger.error(f"MongoDB connection failed: {e}")
        # In a real app, you might want to exit or retry here
        raise HTTPException(status_code=500, detail="Could not connect to database. Check MONGO_URI and network access.")
    except Exception as e:
        logger.error(f"Failed to load NLP model or other startup error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Backend startup failed.")

@app.on_event("shutdown")
async def shutdown_db_client():
    """
    Closes the MongoDB connection on application shutdown.
    """
    if client:
        client.close()
        logger.info("MongoDB connection closed.")

class ChatQuery(BaseModel):
    user_id: str
    query_text: str
    language: str = "en" # Default to English,'hi'-Hindii obv , 'hinglish'

class ChatResponse(BaseModel):
    bot_response: str
    status: str 
    language: str
    query_id: str = None 

class LogEntry(BaseModel):
    timestamp: datetime
    user_id: str
    query_text: str
    bot_response_text: str
    status: str
    language: str
    similarity_score: float = None


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
        logger.error(f"Error generating embedding for query '{user_query_text}': {e}", exc_info=True)
        #a Classic generic response
        await log_user_interaction(
            user_id=user_id,
            query_text=user_query_text,
            bot_response_text="Error processing your query. Please try again.",
            status="error",
            language=language
        )
        raise HTTPException(status_code=500, detail="Error processing your query.")

    # Week 3

    bot_response_text = "I'm still learning! For now, I can only acknowledge your query."
    status = "unanswered" # Default to unanswered for now

    await log_user_interaction(
        user_id=user_id,
        query_text=user_query_text,
        bot_response_text=bot_response_text,
        status=status,
        language=language,
        
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
    similarity_score: float = None 
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
        
        result = db[LOGS_COLLECTION].insert_one(log_entry.dict())
        logger.info(f"Logged interaction with ID: {result.inserted_id}")
        return {"message": "Log entry created successfully", "id": str(result.inserted_id)}
    except Exception as e:
        logger.error(f"Failed to log interaction: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to log interaction.")

# Admin Dashboard Endpoints (Basic - Week 3?)

@app.get("/admin/unanswered_queries")
async def get_unanswered_queries():
    """Retrieves all unanswered queries for admin review."""
    if not db:
        raise HTTPException(status_code=503, detail="Database not connected.")
    try:
        queries = list(db[LOGS_COLLECTION].find({"status": "unanswered"}))
        for q in queries:
            if "_id" in q: 
                q["_id"] = str(q["_id"])
        return queries
    except Exception as e:
        logger.error(f"Failed to retrieve unanswered queries: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve data.")

@app.get("/admin/logs")
async def get_all_logs():
    """Retrieves all interaction logs."""
    if not db:
        raise HTTPException(status_code=503, detail="Database not connected.")
    try:
        
        logs = list(db[LOGS_COLLECTION].find({}))
        for l in logs:
            if "_id" in l: 
                l["_id"] = str(l["_id"])
        return logs
    except Exception as e:
        logger.error(f"Failed to retrieve logs: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve data.")