import pandas as pd
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from sentence_transformers import SentenceTransformer
import os
import logging
from dotenv import load_dotenv
from typing import List, Dict, Any

# --- Configuration ---
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '..', 'backend', '.env'))

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME", "chatbot_db")
FAQ_COLLECTION = os.getenv("FAQ_COLLECTION", "faqs")
NLP_MODEL_NAME = os.getenv("NLP_MODEL_NAME", "paraphrase-multilingual-MiniLM-L12-v2")

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_etl(faq_data_path: str):
    """
    Runs the ETL pipeline to ingest FAQ data into MongoDB.
    This version correctly processes a CSV with consolidated FAQ data.
    """
    logger.info("Starting ETL pipeline...")
    client = None
    model = None
    try:
        # --- 1. Connect to MongoDB ---
        logger.info("Connecting to MongoDB for ETL...")
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        client.admin.command('ping')
        db = client[DB_NAME]
        faqs_collection = db[FAQ_COLLECTION]
        logger.info("Connected to MongoDB for ETL.")

        # --- 2. Load NLP Model ---
        logger.info(f"Loading NLP model for ETL: {NLP_MODEL_NAME}...")
        model = SentenceTransformer(NLP_MODEL_NAME)
        logger.info("NLP model loaded for ETL.")

        # --- 3. Extract Data ---
        logger.info(f"Extracting data from {faq_data_path}...")
        df = pd.read_csv(faq_data_path, keep_default_na=False)
        logger.info(f"Extracted {len(df)} rows from CSV.")

        # --- 4. Transform Data (Generate Embeddings & Clean) ---
        processed_faqs = []
        for _, row in df.iterrows():
            faq_doc: Dict[str, Any] = {
                "question_id": str(row['ID']),
                "category": str(row['category']).strip(),
                "answer_en": str(row['answer_en']).strip(),
                "answer_hi": str(row['answer_hi']).strip(),
                "keywords_en": [k.strip() for k in str(row['keywords_en']).split(',') if k.strip()],
                "keywords_hi": [k.strip() for k in str(row['keywords_hi']).split(',') if k.strip()],
                "embedding": []
            }

            text_for_embedding = ""
            if faq_doc['answer_en']:
                text_for_embedding += faq_doc['answer_en'] + " "
            if faq_doc['answer_hi']:
                text_for_embedding += faq_doc['answer_hi'] + " "
            if faq_doc['keywords_en']:
                text_for_embedding += " ".join(faq_doc['keywords_en']) + " "
            if faq_doc['keywords_hi']:
                text_for_embedding += " ".join(faq_doc['keywords_hi'])

            text_for_embedding = text_for_embedding.strip()

            if not text_for_embedding:
                logger.warning(f"Skipping FAQ ID {faq_doc['question_id']} due to missing text for embedding.")
                continue

            faq_doc['embedding'] = model.encode(text_for_embedding, convert_to_tensor=False).tolist()
            processed_faqs.append(faq_doc)
        
        logger.info(f"Transformed {len(processed_faqs)} unique FAQs with embeddings.")

        # --- 5. Load Data ---
        if processed_faqs:
            faqs_collection.delete_many({})
            faqs_collection.insert_many(processed_faqs)
            logger.info(f"Successfully loaded {len(processed_faqs)} FAQs into MongoDB.")
        else:
            logger.warning("No FAQs to load after processing.")

    except FileNotFoundError:
        logger.error(f"FAQ data file not found at: {faq_data_path}")
    except ConnectionFailure as e:
        logger.error(f"MongoDB connection failed during ETL: {e}")
    except Exception as e:
        logger.error(f"An error occurred during ETL: {e}", exc_info=True)
    finally:
        if client:
            client.close()
            logger.info("MongoDB connection closed for ETL.")

if __name__ == "__main__":
    faq_data_file = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'faqs_raw.csv')
    
    if not os.path.exists(faq_data_file):
        logger.error(f"FAQ data file not found: {faq_data_file}")
        logger.info("Please ensure 'faqs_raw.csv' is in the 'data/' directory at the project root.")
    
    run_etl(faq_data_file)