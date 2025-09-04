
import os
import json
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from sentence_transformers import SentenceTransformer
import logging
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.env')))

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("LEGAL_DB", "legal_db")  # Changed to use LEGAL_DB
SBERT_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
DATA_DIRECTORY = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    sbert_model = SentenceTransformer(SBERT_MODEL_NAME)
    logger.info("Sentence-Transformer model loaded for multilingual embeddings.")
except Exception as e:
    logger.error(f"Failed to load Sentence-Transformer model: {e}")
    sbert_model = None

def ingest_file_to_mongodb(file_path: str, client: MongoClient):
    """
    Ingests a single JSON file into a MongoDB collection, generating embeddings.
    The collection name is derived from the filename.
    """
    if not sbert_model:
        logger.error("Embedding model not loaded. Cannot ingest file.")
        return

    collection_name = os.path.basename(file_path).replace('.json', '').replace('.', '_')
    logger.info(f"Processing file: '{os.path.basename(file_path)}' into collection '{collection_name}'")

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        
        if not isinstance(chunks, list) or not all(isinstance(c, dict) for c in chunks):
            logger.error(f"Skipping {file_path}: content is not a list of dictionaries.")
            return

        db = client[DB_NAME]
        collection = db[collection_name]
        
        processed_count = 0
        for i, chunk in enumerate(chunks):
            content_en = chunk.get("content_en", "").strip()
            content_hi = chunk.get("content_hi", "").strip()

            if not content_en and not content_hi:
                logger.warning(f"Skipping chunk {i} in {file_path}: both 'content_en' and 'content_hi' are empty.")
                continue

            try:
                # Generate embeddings for non-empty content
                if content_en:
                    chunk['embedding_en'] = sbert_model.encode(content_en).tolist()
                if content_hi:
                    chunk['embedding_hi'] = sbert_model.encode(content_hi).tolist()

                # Use content and source to uniquely identify a chunk for upserting
                filter_query = {
                    "source": chunk.get("source"),
                    "content_en": content_en,
                    "content_hi": content_hi
                }
                
                collection.update_one(
                    filter_query,
                    {"$set": chunk},
                    upsert=True
                )
                processed_count += 1

            except Exception as e:
                logger.error(f"Error processing chunk {i} in {file_path}: {e}", exc_info=True)
        
        logger.info(f"Successfully ingested/updated {processed_count} documents from {file_path} into collection '{collection_name}'.")

    except json.JSONDecodeError:
        logger.error(f"Invalid JSON format in file: {file_path}")
    except Exception as e:
        logger.error(f"An unexpected error occurred for file {file_path}: {e}", exc_info=True)

def main():
    """
    Main function to walk through the data directory and ingest all JSON files.
    """
    if not sbert_model:
        logger.error("Embedding model is not available. Aborting ingestion process.")
        return
        
    if not os.path.exists(DATA_DIRECTORY):
        logger.error(f"Data directory not found: {DATA_DIRECTORY}")
        return

    logger.info(f"Starting ingestion process from directory: {DATA_DIRECTORY}")
    
    try:
        with MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000) as client:
            client.admin.command('ping') # Verify connection
            logger.info("Successfully connected to MongoDB.")
            
            for root, _, files in os.walk(DATA_DIRECTORY):
                for file in files:
                    if file.endswith('.json'):
                        file_path = os.path.join(root, file)
                        ingest_file_to_mongodb(file_path, client)

    except ConnectionFailure as e:
        logger.error(f"MongoDB connection failed: {e}")
    except Exception as e:
        logger.error(f"An error occurred during the ingestion process: {e}", exc_info=True)
    
    logger.info("Ingestion process finished.")

if __name__ == "__main__":
    main()