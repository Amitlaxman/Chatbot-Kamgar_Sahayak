import logging
import os
from typing import List, Dict, Any, Optional
from pymongo import MongoClient, ASCENDING
from pymongo.errors import ConnectionFailure
from passlib.context import CryptContext
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# MongoDB client and database instances
client: Optional[MongoClient] = None
db = None

# Collection names from environment variables
LOGS_COLLECTION = os.getenv('LOGS_COLLECTION', 'logs')
KEYWORDS_COLLECTION = os.getenv('KEYWORDS_COLLECTION', 'keywords')
ADMIN_USERS_COLLECTION = os.getenv('ADMIN_USERS_COLLECTION', 'admin_users')
LEGAL_CONTENT_PREFIX = 'legal_content_'  # Prefix for all legal content collections

async def connect_to_mongo(mongo_uri: str, db_name: str = "chatbot_db") -> None:
    """Initialize MongoDB connection."""
    global client, db
    try:
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        # Test connection
        client.admin.command('ping')
        
        # Initialize database
        db = client[db_name]
        
        # Drop existing indexes to avoid conflicts
        try:
            db.users.drop_indexes()
            db[KEYWORDS_COLLECTION].drop_indexes()
            db[LOGS_COLLECTION].drop_indexes()
            db[ADMIN_USERS_COLLECTION].drop_indexes()
            logger.info("Dropped existing indexes")
        except Exception as e:
            logger.warning(f"Error dropping indexes: {e}")

        # Clean up null values
        try:
            db[KEYWORDS_COLLECTION].delete_many({"keyword": None})
            db.users.delete_many({"email": None})
            db[ADMIN_USERS_COLLECTION].delete_many({"email": None})
            logger.info("Cleaned up null values")
        except Exception as e:
            logger.warning(f"Error cleaning up null values: {e}")

        # Create new indexes with explicit names
        db.users.create_index(
            [("email", ASCENDING)],
            unique=True,
            partialFilterExpression={"email": {"$type": "string"}},
            name="unique_email_users"
        )
        
        db[LOGS_COLLECTION].create_index(
            [("timestamp", ASCENDING)],
            name="timestamp_logs"
        )
        
        db[KEYWORDS_COLLECTION].create_index(
            [("keyword", ASCENDING)],
            unique=True,
            partialFilterExpression={"keyword": {"$type": "string"}},
            name="unique_keyword"
        )
        
        db[ADMIN_USERS_COLLECTION].create_index(
            [("email", ASCENDING)],
            unique=True,
            partialFilterExpression={"email": {"$type": "string"}},
            name="unique_email_admins"
        )
        
        logger.info(f"✅ Connected to MongoDB database: {db_name} and created new indexes")
        
    except ConnectionFailure as e:
        logger.error(f"❌ MongoDB connection failed: {e}")
        raise
    except Exception as e:
        logger.error(f"❌ Unexpected error connecting to MongoDB: {e}", exc_info=True)
        raise

async def migrate_collections() -> None:
    """Organize collections in the single database."""
    try:
        collections = db.list_collection_names()
        
        # Create necessary indexes for main collections
        db.users.create_index([("email", ASCENDING)], unique=True)
        db[LOGS_COLLECTION].create_index([("timestamp", ASCENDING)])
        db[KEYWORDS_COLLECTION].create_index([("keyword", ASCENDING)], unique=True)
        db[ADMIN_USERS_COLLECTION].create_index([("email", ASCENDING)], unique=True)
        
        logger.info("Migration completed successfully")

    except Exception as e:
        logger.error(f"Error during collection organization: {e}")
        raise

async def close_mongo_connection():
    global client
    if client:
        client.close()
        logger.info("🔌 MongoDB connection closed.")

def get_mongo_db():
    """Get the MongoDB database connection."""
    if db is None:
        raise ConnectionFailure("Database connection not established")
    return db

def get_mongo_client() -> MongoClient:
    """Get the MongoDB client connection."""
    if client is None:
        raise ConnectionFailure("MongoDB client not established")
    return client

def get_all_faqs() -> List[Dict[str, Any]]:
    """Get all FAQ documents from the database."""
    mongodb = get_mongo_db()
    # Only get content collections (exclude system and utility collections)
    collections = [c for c in mongodb.list_collection_names() 
                  if not c.startswith('system.') 
                  and c not in {LOGS_COLLECTION, 'users', KEYWORDS_COLLECTION, ADMIN_USERS_COLLECTION}]
    
    results = []
    for cname in collections:
        docs = list(mongodb[cname].find({}, {"_id": 0}))
        results.extend(docs)
    logger.info(f"Fetched {len(results)} documents from {len(collections)} collections")
    return results

async def insert_log_entry(entry: Dict[str, Any]) -> str:
    """Insert a log entry into the logs collection."""
    result = get_mongo_db()[LOGS_COLLECTION].insert_one(entry)
    logger.info(f"📝 Log entry inserted with ID: {result.inserted_id}")
    return str(result.inserted_id)

def get_unanswered_logs() -> List[Dict[str, Any]]:
    """Get all unanswered log entries."""
    logs = get_mongo_db()[LOGS_COLLECTION].find({
        "$or": [
            {"answer": None},
            {"answer": ""},
            {"answer": {"$exists": False}}
        ]
    })
    return [{**log, "_id": str(log["_id"])} for log in logs]

def get_all_logs_entries() -> List[Dict[str, Any]]:
    """Get all log entries."""
    logs = get_mongo_db()[LOGS_COLLECTION].find({})
    return [{**log, "_id": str(log["_id"])} for log in logs]

async def create_user(user_data: Dict[str, Any]) -> str:
    """Create a new user in the database."""
    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    mongodb = get_mongo_db()
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
    return get_mongo_db().users.find_one({"email": email})

async def verify_user(email: str, password: str) -> bool:
    """Verify user credentials."""
    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    user = await get_user_by_email(email)
    if not user:
        return False
    return pwd_context.verify(password, user.get("hashed_password", ""))

async def get_admin_user(email: str) -> Optional[Dict[str, Any]]:
    """Get admin user by email."""
    return get_mongo_db()[ADMIN_USERS_COLLECTION].find_one({"email": email})

async def create_admin_user(admin_data: Dict[str, Any]) -> str:
    """Create a new admin user."""
    result = get_mongo_db()[ADMIN_USERS_COLLECTION].insert_one(admin_data)
    return str(result.inserted_id)

async def insert_admin_marking(log_id: str, marking: Dict[str, Any]) -> bool:
    """Insert admin marking for a log entry."""
    try:
        result = get_mongo_db()[LOGS_COLLECTION].update_one(
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
        result = get_mongo_db()[LOGS_COLLECTION].update_one(
            {"_id": log_id},
            {"$set": {"answer": answer}}
        )
        return result.modified_count > 0
    except Exception as e:
        logger.error(f"Error inserting admin answer: {e}")
        return False