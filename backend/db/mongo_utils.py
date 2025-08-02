from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# Global client and database instances
client: Optional[MongoClient] = None
db = None

async def connect_to_mongo(mongo_uri: str, db_name: str):

    global client, db
    try:
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        client.admin.command('ping') # Test connection
        db = client[db_name]
        logger.info("Successfully connected to MongoDB.")
    except ConnectionFailure as e:
        logger.error(f"MongoDB connection failed: {e}")
        raise ConnectionFailure(f"Could not connect to MongoDB: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during MongoDB connection: {e}", exc_info=True)
        raise Exception(f"MongoDB connection error: {e}")

async def close_mongo_connection():
    
    global client
    if client:
        client.close()
        logger.info("MongoDB connection closed.")

def get_mongo_db():
    
    if db is None:
        raise ConnectionFailure("MongoDB connection not established.")
    return db

# --- CRUD Operations for FAQs ---
async def get_all_faqs() -> List[Dict[str, Any]]:
    '''Retrieves all FAQs from the database.'''
    try:
        faqs_collection = get_mongo_db()["faqs"]
        # Use list() to convert cursor to list for synchronous pymongo
        return list(faqs_collection.find({}))
    except Exception as e:
        logger.error(f"Error retrieving all FAQs: {e}", exc_info=True)
        raise

async def insert_faqs(faqs: List[Dict[str, Any]]):
    '''Inserts a list of FAQs into the database.'''
    try:
        faqs_collection = get_mongo_db()["faqs"]
        if faqs:
            faqs_collection.insert_many(faqs)
            logger.info(f"Inserted {len(faqs)} FAQs.")
    except Exception as e:
        logger.error(f"Error inserting FAQs: {e}", exc_info=True)
        raise

async def delete_all_faqs():
    '''Deletes all FAQs from the database.'''
    try:
        faqs_collection = get_mongo_db()["faqs"]
        faqs_collection.delete_many({})
        logger.info("All FAQs deleted.")
    except Exception as e:
        logger.error(f"Error deleting all FAQs: {e}", exc_info=True)
        raise

'''Logs'''
async def insert_log_entry(log_data: Dict[str, Any]):
    try:
        logs_collection = get_mongo_db()["logs"]
        result = logs_collection.insert_one(log_data)
        logger.info(f"Log entry inserted with ID: {result.inserted_id}")
        return str(result.inserted_id)
    except Exception as e:
        logger.error(f"Error inserting log entry: {e}", exc_info=True)
        raise

async def get_unanswered_logs() -> List[Dict[str, Any]]:
    try:
        logs_collection = get_mongo_db()["logs"]
        # Convert ObjectId to string for JSON serialization
        logs = list(logs_collection.find({"status": "unanswered"}))
        for log in logs:
            if "_id" in log:
                log["_id"] = str(log["_id"])
        return logs
    except Exception as e:
        logger.error(f"Error retrieving unanswered logs: {e}", exc_info=True)
        raise

async def get_all_logs_entries() -> List[Dict[str, Any]]:
    '''Retrieves all log entries.'''
    try:
        logs_collection = get_mongo_db()["logs"]
        logs = list(logs_collection.find({}))
        for log in logs:
            if "_id" in log:
                log["_id"] = str(log["_id"])
        return logs
    except Exception as e:
        logger.error(f"Error retrieving all logs: {e}", exc_info=True)
        raise

# - Admin Users ---
async def get_admin_user(username: str) -> Optional[Dict[str, Any]]:
    '''Retrieves an admin user by username.'''
    try:
        admin_collection = get_mongo_db()["admin_users"]
        user = admin_collection.find_one({"username": username})
        return user
    except Exception as e:
        logger.error(f"Error retrieving admin user {username}: {e}", exc_info=True)
        raise

async def create_admin_user(user_data: Dict[str, Any]):
    '''Creates a new admin user.'''
    try:
        admin_collection = get_mongo_db()["admin_users"]
        result = admin_collection.insert_one(user_data)
        logger.info(f"Admin user created with ID: {result.inserted_id}")
        return str(result.inserted_id)
    except Exception as e:
        logger.error(f"Error creating admin user: {e}", exc_info=True)
        raise