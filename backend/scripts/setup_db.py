import asyncio
import os
import logging
from dotenv import load_dotenv
from ..db.mongo_utils import connect_to_mongo, migrate_collections

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def setup_databases():
    """Set up and organize MongoDB databases."""
    try:
        # Load environment variables
        load_dotenv()
        mongo_uri = os.getenv('MONGO_URI')
        legal_db = os.getenv('LEGAL_DB', 'legal_db')
        chatbot_db = os.getenv('CHATBOT_DB', 'chatbot_db')
        admin_db = os.getenv('ADMIN_DB', 'admin_db')
        
        if not mongo_uri:
            raise ValueError("MONGO_URI not found in environment variables")
        
        # Connect to MongoDB
        logger.info("Connecting to MongoDB...")
        await connect_to_mongo(mongo_uri, legal_db, chatbot_db, admin_db)
        
        # Migrate collections to their proper databases
        logger.info("Starting collection migration...")
        await migrate_collections()
        
        logger.info("Database setup completed successfully!")

    except Exception as e:
        logger.error(f"Error during database setup: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(setup_databases())