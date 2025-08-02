
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
async def read_root():
    """Basic root endpoint to confirm API is running."""
    return {"message": "Shramik Saathi Chatbot Backend is running!"}

