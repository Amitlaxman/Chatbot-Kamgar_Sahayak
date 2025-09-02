
import os
import json
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from sentence_transformers import SentenceTransformer
import logging
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.env')))

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = "legal_db"
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

def ingest_file_to_mongodb(file_path: str):
    if not sbert_model:
        logger.error("Embedding model not loaded. Skipping ingestion.")
        return
    try:
        collection_name = os.path.basename(file_path).replace('.json', '')
        with open(file_path, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        if not isinstance(chunks, list) or not all(isinstance(c, dict) for c in chunks):
            logger.error(f"File {file_path} does not contain a valid list of chunk dictionaries. Skipping.")
            return
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        db = client[DB_NAME]
        collection = db[collection_name]
        logger.info(f"Connected to collection '{collection_name}' in database '{DB_NAME}'.")
        processed_count = 0
        for chunk in chunks:
            if "content_en" in chunk and "content_hi" in chunk:
                try:
                    chunk['embedding_en'] = sbert_model.encode(chunk["content_en"]).tolist()
                    chunk['embedding_hi'] = sbert_model.encode(chunk["content_hi"]).tolist()
                    collection.update_one(
                        {"source": chunk.get("source"), "state": chunk.get("state")},
                        {"$set": chunk},
                        upsert=True
                    )
                    processed_count += 1
                except Exception as embed_e:
                    logger.error(f"Error generating embeddings for a chunk in {file_path}: {embed_e}. Skipping chunk.")
            else:
                logger.warning(f"Skipping chunk in {file_path} due to missing 'content_en' or 'content_hi' field.")
        logger.info(f"Successfully ingested {processed_count} documents from {file_path}.")
        client.close()
        logger.info("MongoDB connection closed.")
    except ConnectionFailure as e:
        logger.error(f"MongoDB connection failed for file {file_path}: {e}")
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON format in file: {file_path}")
    except Exception as e:
        logger.error(f"An unexpected error occurred for file {file_path}: {e}", exc_info=True)

def main():
    if not os.path.exists(DATA_DIRECTORY):
        logger.error(f"Data directory not found: {DATA_DIRECTORY}")
        return
    logger.info(f"Starting ingestion process from directory: {DATA_DIRECTORY}")
    for root, dirs, files in os.walk(DATA_DIRECTORY):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                ingest_file_to_mongodb(file_path)

if __name__ == "__main__":
    main()