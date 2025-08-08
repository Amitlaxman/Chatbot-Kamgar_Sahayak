from backend.routes import chat_routes, admin_routes
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
import os
import logging
import jwt
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from passlib.context import CryptContext

from backend.db.mongo_utils import connect_to_mongo, close_mongo_connection, get_admin_user
from backend.nlp.model_loader import load_nlp_model


# Load environment variables

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '.env'))

# Configuration
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
print("MONGO_URI:", MONGO_URI)  # Debug print to verify the value
DB_NAME = os.getenv("DB_NAME", "chatbot_db")
NLP_MODEL_NAME = os.getenv("NLP_MODEL_NAME", "paraphrase-multilingual-MiniLM-L12-v2")
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-super-secret-key-please-change-this-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# FastAPI app initialization
app = FastAPI(
    title="Shramik Saathi Chatbot Backend",
    description="Backend API for the multilingual chatbot assisting laborers in MP, India",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS configuration
origins = [
    "http://localhost:3000",  # React default dev server
    "http://127.0.0.1:3000",
    # Add your production frontend URL here
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Security setup
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/admin_api/token")



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
async def read_root():
    """Basic root endpoint to confirm API is running."""
    return {"message": "Shramik Saathi Chatbot Backend is running!"}
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
from backend.routes import admin_routes as admin_routes, chat_routes
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

# Security utility functions
def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """Generate password hash."""
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token."""
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_admin_user(token: str = Depends(oauth2_scheme)) -> dict:
    """Validate JWT token and return admin user."""
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
        user = await get_admin_user(username)
        if user is None:
            raise credentials_exception
        return user
    except jwt.PyJWTError:
        raise credentials_exception

# Application startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize database connection and NLP model."""
    try:
        logger.info("Starting up backend services...")
        await connect_to_mongo(MONGO_URI, DB_NAME)
        await load_nlp_model(NLP_MODEL_NAME)
        logger.info("Backend services started successfully")
    except Exception as e:
        logger.error(f"Startup failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Backend failed to start up")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown."""
    logger.info("Shutting down backend services...")
    await close_mongo_connection()
    logger.info("Backend services shut down successfully")
# Include routers
app.include_router(chat_routes.router, prefix="/chat_api", tags=["Chatbot"])
app.include_router(admin_routes.router, prefix="/admin_api", tags=["Admin"])

# Root endpoint
@app.get("/")
async def read_root():
    """Root endpoint to confirm API is running."""
    return {
        "message": "Shramik Saathi Chatbot Backend is running!",
        "version": "0.1.0",
        "docs_url": "/docs",
        "redoc_url": "/redoc"
    }

