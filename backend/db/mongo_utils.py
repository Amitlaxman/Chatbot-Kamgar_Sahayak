import logging
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
from bson import ObjectId
from pymongo import MongoClient, ASCENDING
from pymongo.errors import ConnectionFailure
from passlib.context import CryptContext
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# MongoDB client and database instances
client: Optional[MongoClient] = None
legal_db = None     # For RAG content/chunks
chatbot_db = None   # For users, logs, keywords
admin_db = None     # For admin users, markings, answers

# Collection names from environment variables
LOGS_COLLECTION = os.getenv('LOGS_COLLECTION', 'logs')
KEYWORDS_COLLECTION = os.getenv('KEYWORDS_COLLECTION', 'keywords')
ADMIN_USERS_COLLECTION = os.getenv('ADMIN_USERS_COLLECTION', 'admin_users')
ADMIN_MARKINGS_COLLECTION = 'admin_markings'
ADMIN_ANSWERS_COLLECTION = 'admin_answers'

async def connect_to_mongo(mongo_uri: str) -> None:
    """Initialize MongoDB connections for both legal and chatbot databases."""
    global client, legal_db, chatbot_db, admin_db
    try:
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        # Test connection
        client.admin.command('ping')
        
        # Initialize separate databases
        legal_db = client['legal_db']
        chatbot_db = client['chatbot_db']
        admin_db = client['admin_db'] # <--- NEW LINE ADDED HERE
        
        # Drop existing indexes in chatbot_db to avoid conflicts
        try:
            chatbot_db.users.drop_indexes()
            chatbot_db[KEYWORDS_COLLECTION].drop_indexes()
            chatbot_db[LOGS_COLLECTION].drop_indexes()
            chatbot_db[ADMIN_USERS_COLLECTION].drop_indexes()
            logger.info("Dropped existing indexes in chatbot_db")
        except Exception as e:
            logger.warning(f"Error dropping indexes: {e}")

        # Clean up null values in chatbot_db
        try:
            chatbot_db[KEYWORDS_COLLECTION].delete_many({"keyword": None})
            chatbot_db.users.delete_many({"email": None})
            chatbot_db[ADMIN_USERS_COLLECTION].delete_many({"email": None})
            logger.info("Cleaned up null values in chatbot_db")
        except Exception as e:
            logger.warning(f"Error cleaning up null values: {e}")

        # Create new indexes with explicit names in chatbot_db
        chatbot_db.users.create_index(
            [("email", ASCENDING)],
            unique=True,
            partialFilterExpression={"email": {"$type": "string"}},
            name="unique_email_users"
        )
        
        chatbot_db[LOGS_COLLECTION].create_index(
            [("timestamp", ASCENDING)],
            name="timestamp_logs"
        )
        
        chatbot_db[KEYWORDS_COLLECTION].create_index(
            [("keyword", ASCENDING)],
            unique=True,
            partialFilterExpression={"keyword": {"$type": "string"}},
            name="unique_keyword"
        )
        
        chatbot_db[ADMIN_USERS_COLLECTION].create_index(
            [("email", ASCENDING)],
            unique=True,
            partialFilterExpression={"email": {"$type": "string"}},
            name="unique_email_admins"
        )
        
        logger.info("✅ Connected to MongoDB databases: legal_db (content) and chatbot_db (users/logs)")
        
    except ConnectionFailure as e:
        logger.error(f"❌ MongoDB connection failed: {e}")
        raise
    except Exception as e:
        logger.error(f"❌ Unexpected error connecting to MongoDB: {e}", exc_info=True)
        raise

def get_legal_db():
    """Get the legal content database connection."""
    if legal_db is None:
        raise ConnectionFailure("Legal database connection not established")
    return legal_db

def get_chatbot_db():
    """Get the chatbot database connection."""
    if chatbot_db is None:
        raise ConnectionFailure("Chatbot database connection not established")
    return chatbot_db

# <--- THE MISSING FUNCTION DEFINITION IS ADDED HERE
def get_admin_db():
    """Get the admin database connection."""
    if admin_db is None:
        raise ConnectionFailure("Admin database connection not established")
    return admin_db
# ---> END OF ADDED CODE

def get_all_faqs() -> List[Dict[str, Any]]:
    """Get all FAQ documents from the legal database."""
    db = get_legal_db()
    # Get all collections except system collections
    collections = [c for c in db.list_collection_names() 
                   if not c.startswith('system.')]
    
    results = []
    for cname in collections:
        docs = list(db[cname].find({}, {"_id": 0}))
        results.extend(docs)
    logger.info(f"Fetched {len(results)} documents from {len(collections)} legal content collections")
    return results

async def insert_log_entry(entry: Dict[str, Any]) -> str:
    """Insert a log entry into the logs collection in chatbot_db."""
    result = get_chatbot_db()[LOGS_COLLECTION].insert_one(entry)
    logger.info(f"📝 Log entry inserted with ID: {result.inserted_id}")
    return str(result.inserted_id)

def get_unanswered_logs() -> List[Dict[str, Any]]:
    """Get all unanswered log entries from chatbot_db."""
    logs = get_chatbot_db()[LOGS_COLLECTION].find({
        "$or": [
            {"answer": None},
            {"answer": ""},
            {"answer": {"$exists": False}}
        ]
    })
    return [{**log, "_id": str(log["_id"])} for log in logs]


def get_all_logs_entries() -> List[Dict[str, Any]]:
    """Get all log entries."""
    logs = get_chatbot_db()[LOGS_COLLECTION].find({})
    return [{**log, "_id": str(log["_id"])} for log in logs]

async def create_user(user_data: Dict[str, Any]) -> str:
    """Create a new user in the database."""
    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    mongodb = get_chatbot_db()
    users_collection = mongodb.users

    if users_collection.find_one({"email": user_data["email"]}):
        raise ValueError("Email already registered")

    if "password" in user_data:
        plain_pw = user_data.pop("password")
        user_data["hashed_password"] = pwd_context.hash(plain_pw)
    elif "hashed_password" in user_data:
        user_data["hashed_password"] = str(user_data["hashed_password"])
    else:
        raise ValueError("Password is required to create a user")

    result = users_collection.insert_one(user_data)
    return str(result.inserted_id)

async def get_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    """Get user by email."""
    return get_chatbot_db().users.find_one({"email": email})

async def verify_user(email: str, password: str) -> bool:
    """Verify user credentials."""
    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    user = await get_user_by_email(email)
    if not user:
        return False
    return pwd_context.verify(password, user.get("hashed_password", ""))

async def get_admin_user(email: str) -> Optional[Dict[str, Any]]:
    """Get admin user by email."""
    return get_admin_db()[ADMIN_USERS_COLLECTION].find_one({"email": email})

async def create_admin_user(admin_data: Dict[str, Any]) -> str:
    """Create a new admin user."""
    result = get_admin_db()[ADMIN_USERS_COLLECTION].insert_one(admin_data)
    return str(result.inserted_id)

async def insert_admin_marking(log_id: str, marking: Dict[str, Any]) -> bool:
    """Insert admin marking for a log entry."""
    try:
        result = get_admin_db()[LOGS_COLLECTION].update_one(
            {"_id": log_id},
            {"$set": marking}
        )
        return result.modified_count > 0
    except Exception as e:
        logger.error(f"Error inserting admin marking: {e}")
        return False

async def insert_admin_answer(log_id: str, answer: str) -> bool:
    """Insert admin answer for a log entry."""
    try:
        result = get_admin_db()[LOGS_COLLECTION].update_one(
            {"_id": log_id},
            {"$set": {"answer": answer}}
        )
        return result.modified_count > 0
    except Exception as e:
        logger.error(f"Error inserting admin answer: {e}")
        return False