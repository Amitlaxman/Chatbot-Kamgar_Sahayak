
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import logging
from dotenv import load_dotenv
from backend.db.mongo_utils import connect_to_mongo, close_mongo_connection
from backend.nlp.model_loader import load_nlp_model
from backend.routes import chat_routes, admin_routes 


load_dotenv() 

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME", "chatbot_db")
NLP_MODEL_NAME = os.getenv("NLP_MODEL_NAME", "paraphrase-multilingual-MiniLM-L12-v2")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Shramik Saathi Chatbot Backend",
    description="Backend API for the multilingual chatbot assisting laborers in MP, India.",
    version="0.1.0"
)

# --- CORS Middleware ---
origins = [
    "http://localhost:3000", # React default dev server
    "http://127.0.0.1:3000",

]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # Allows all methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"], # Allows all headers
)



@app.on_event("startup")
async def startup_event():
    """
    Connects to MongoDB and loads the NLP model on application startup.
    """
    try:
        logger.info("Starting up backend services...")
        # Connect to MongoDB
        await connect_to_mongo(MONGO_URI, DB_NAME)
        logger.info("MongoDB connection established.")

        # Load NLP model
        await load_nlp_model(NLP_MODEL_NAME)
        logger.info("NLP model loaded successfully.")

    except HTTPException as e:
        logger.error(f"Startup failed due to HTTP error: {e.detail}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred during startup: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Backend failed to start up.")

@app.on_event("shutdown")
async def shutdown_event():
    """
    Closes the MongoDB connection on application shutdown.
    """
    logger.info("Shutting down backend services...")
    await close_mongo_connection()
    logger.info("MongoDB connection closed.")


app.include_router(chat_routes.router, prefix="/chat_api", tags=["Chatbot"])
app.include_router(admin_routes.router, prefix="/admin_api", tags=["Admin"])

@app.get("/")
async def read_root():from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, Field
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from sentence_transformers import SentenceTransformer
from passlib.context import CryptContext
import jwt
import os
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import numpy as np # For efficient cosine similarity, dependency of sentence_transformers

# --- Configuration & Environment Variables ---
# Load environment variables from .env file for local development
# Make sure you have a .env file in the 'backend' directory
# with MONGO_URI, JWT_SECRET_KEY, etc.
from dotenv import load_dotenv
load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = os.getenv("DB_NAME", "chatbot_db")
FAQ_COLLECTION = os.getenv("FAQ_COLLECTION", "faqs")
LOGS_COLLECTION = os.getenv("LOGS_COLLECTION", "logs")
ADMIN_USERS_COLLECTION = os.getenv("ADMIN_USERS_COLLECTION", "admin_users")
NLP_MODEL_NAME = os.getenv("NLP_MODEL_NAME", "paraphrase-multilingual-MiniLM-L12-v2") # Good for multilingual

# Chatbot Logic Configuration
CONFIDENCE_THRESHOLD = 0.75 # If similarity is below this, query is "unanswered"

# Admin Security Configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-super-secret-key-please-change-this-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Shramik Saathi Chatbot Backend (Demo)",
    description="Self-contained Backend API for the multilingual chatbot demo.",
    version="0.1.0"
)

# --- CORS Middleware ---
origins = [
    "http://localhost:3000", # React default dev server
    "http://127.0.0.1:3000",
    # Add your deployed frontend URL here when you deploy
    # "https://your-frontend-app.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Global Instances ---
client: Optional[MongoClient] = None
db = None
nlp_model: Optional[SentenceTransformer] = None

# --- Security Utilities (Admin) ---
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/admin_api/token") # Note: tokenUrl is relative to app root

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_admin_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        # In a real app, you'd fetch user from DB to verify existence/role
        # For this demo, we'll assume valid token means valid user
        user = await get_admin_user_from_db(username) # Use inline DB function
        if user is None:
            raise credentials_exception
        return user
    except jwt.PyJWTError:
        raise credentials_exception

# --- Pydantic Models (Inline for Demo) ---
class ChatQuery(BaseModel):
    user_id: str = Field(..., description="Unique identifier for the user.")
    query_text: str = Field(..., min_length=1, description="The text query from the user.")
    language: str = Field("en", description="Language of the query (e.g., 'en', 'hi', 'hinglish').")

class ChatResponse(BaseModel):
    bot_response: str = Field(..., description="The chatbot's response text.")
    status: str = Field(..., description="Status of the query (e.g., 'answered', 'unanswered', 'error').")
    language: str = Field(..., description="Language of the bot's response.")
    query_id: Optional[str] = Field(None, description="Optional ID for the processed query.")
    similarity_score: Optional[float] = Field(None, description="Cosine similarity score if answered by NLP.")

class LogEntry(BaseModel):
    timestamp: datetime = Field(default_factory=datetime.now, description="Timestamp of the interaction.")
    user_id: str = Field(..., description="User ID associated with the interaction.")
    query_text: str = Field(..., description="The original query text from the user.")
    bot_response_text: str = Field(..., description="The bot's response text.")
    status: str = Field(..., description="Status of the interaction (e.g., 'answered', 'unanswered', 'error').")
    language: str = Field(..., description="Language of the interaction.")
    similarity_score: Optional[float] = Field(None, description="Similarity score of the match, if applicable.")

class AdminUser(BaseModel): # For storing in DB
    username: str
    hashed_password: str
    role: str # e.g., "admin", "viewer"

class AdminLogin(BaseModel): # For login requests
    username: str
    password: str

# --- MongoDB Utility Functions (Inline for Demo) ---
# Simplified versions for direct use in main.py
async def connect_to_mongo_inline(mongo_uri: str, db_name: str):
    global client, db
    try:
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        client.admin.command('ping')
        db = client[db_name]
        logger.info("MongoDB connection established (inline).")
    except ConnectionFailure as e:
        logger.error(f"MongoDB connection failed (inline): {e}")
        raise HTTPException(status_code=500, detail=f"Database connection error: {e}")
    except Exception as e:
        logger.error(f"Unexpected error during MongoDB connection (inline): {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Database connection error.")

async def close_mongo_connection_inline():
    global client
    if client:
        client.close()
        logger.info("MongoDB connection closed (inline).")

def get_mongo_db_inline():
    if db is None:
        raise HTTPException(status_code=503, detail="MongoDB connection not established.")
    return db

async def get_all_faqs_inline() -> List[Dict[str, Any]]:
    try:
        faqs_collection = get_mongo_db_inline()[FAQ_COLLECTION]
        return list(faqs_collection.find({}))
    except Exception as e:
        logger.error(f"Error retrieving all FAQs (inline): {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve FAQs.")

async def insert_faqs_inline(faqs: List[Dict[str, Any]]):
    try:
        faqs_collection = get_mongo_db_inline()[FAQ_COLLECTION]
        if faqs:
            faqs_collection.insert_many(faqs)
            logger.info(f"Inserted {len(faqs)} FAQs (inline).")
    except Exception as e:
        logger.error(f"Error inserting FAQs (inline): {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to insert FAQs.")

async def insert_log_entry_inline(log_data: Dict[str, Any]):
    try:
        logs_collection = get_mongo_db_inline()[LOGS_COLLECTION]
        result = logs_collection.insert_one(log_data)
        logger.info(f"Log entry inserted with ID: {result.inserted_id} (inline).")
        return str(result.inserted_id)
    except Exception as e:
        logger.error(f"Error inserting log entry (inline): {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to log interaction.")

async def get_unanswered_logs_inline() -> List[Dict[str, Any]]:
    try:
        logs_collection = get_mongo_db_inline()[LOGS_COLLECTION]
        logs = list(logs_collection.find({"status": "unanswered"}))
        for log in logs:
            if "_id" in log: log["_id"] = str(log["_id"])
        return logs
    except Exception as e:
        logger.error(f"Error retrieving unanswered logs (inline): {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve data.")

async def get_all_logs_entries_inline() -> List[Dict[str, Any]]:
    try:
        logs_collection = get_mongo_db_inline()[LOGS_COLLECTION]
        logs = list(logs_collection.find({}))
        for log in logs:
            if "_id" in log: log["_id"] = str(log["_id"])
        return logs
    except Exception as e:
        logger.error(f"Error retrieving all logs (inline): {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve data.")

async def get_admin_user_from_db(username: str) -> Optional[Dict[str, Any]]:
    try:
        admin_collection = get_mongo_db_inline()[ADMIN_USERS_COLLECTION]
        user = admin_collection.find_one({"username": username})
        return user
    except Exception as e:
        logger.error(f"Error retrieving admin user (inline) {username}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve admin user.")

async def create_admin_user_in_db(user_data: Dict[str, Any]):
    try:
        admin_collection = get_mongo_db_inline()[ADMIN_USERS_COLLECTION]
        result = admin_collection.insert_one(user_data)
        logger.info(f"Admin user created with ID: {result.inserted_id} (inline).")
        return 
    """Basic root endpoint to confirm API is running."""
    return {"message": "Shramik Saathi Chatbot Backend is running!"}

