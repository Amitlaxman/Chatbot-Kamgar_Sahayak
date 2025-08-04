import pandas as pd
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from sentence_transformers import SentenceTransformer
import os
import logging
from dotenv import load_dotenv # Import load_dotenv for ETL script too

# Configuration (similar to main.py, but for ETL script)
load_dotenv() # Load .env for ETL script

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = os.getenv("DB_NAME", "chatbot_db")
FAQ_COLLECTION = os.getenv("FAQ_COLLECTION", "faqs")
NLP_MODEL_NAME = os.getenv("NLP_MODEL_NAME", "paraphrase-multilingual-MiniLM-L12-v2")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_etl(faq_data_path: str):
    logger.info("Starting ETL pipeline...")
    client = None
    try:
        # --- 1. Connect to MongoDB ---
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        client.admin.command('ping') # Test connection
        db = client[DB_NAME]
        faqs_collection = db[FAQ_COLLECTION]
        logger.info("Connected to MongoDB for ETL.")

        # --- 2. Load NLP Model ---
        logger.info(f"Loading NLP model for ETL: {NLP_MODEL_NAME}...")
        model = SentenceTransformer(NLP_MODEL_NAME)
        logger.info("NLP model loaded for ETL.")

        # --- 3. Extract Data ---
        logger.info(f"Extracting data from {faq_data_path}...")
        # Assuming CSV for now, but could be JSON, etc.
        df = pd.read_csv(faq_data_path)
        logger.info(f"Extracted {len(df)} FAQs.")

        # --- 4. Transform Data (Generate Embeddings & Clean) ---
        processed_faqs = []
        for index, row in df.iterrows():
            # Ensure these column names match your actual CSV
            question_en = row.get('answer_en')
            question_hi = row.get('answer_hi')
            keywords_en = row.get('keywords_en')
            keywords_hi = row.get('keywords_hi')

            # Combine English and Hindi questions for a more robust embedding if needed,
            # or choose one based on primary language for embedding.
            # For multilingual models, often just the text is enough.
            text_to_embed = ""
            if pd.notna(question_en):
                text_to_embed += str(question_en)
            if pd.notna(question_hi):
                text_to_embed += " " + str(question_hi) # Add space if both exist

            if not text_to_embed.strip(): # Check if combined text is empty
                logger.warning(f"Skipping FAQ {row.get('question_id', index)} due to missing question text for embedding.")
                continue

            # Generate embedding
            embedding = model.encode(text_to_embed.strip(), convert_to_tensor=False).tolist()

            # Prepare document for MongoDB
            faq_doc = {
                "question_id": row.get('question_id', f"faq_{index}"), # Ensure unique ID
                "keywords_en": [k.strip() for k in str(keywords_en).split(',') if k.strip()] if pd.notna(keywords_en) else [],
                "keywords_hi": [k.strip() for k in str(keywords_hi).split(',') if k.strip()] if pd.notna(keywords_hi) else [],
                "answer_en": row.get('answer_en', ''),
                "answer_hi": row.get('answer_hi', ''),
                "category": row.get('category', 'General'),
                "embedding": embedding
            }
            processed_faqs.append(faq_doc)
        logger.info(f"Transformed {len(processed_faqs)} FAQs with embeddings.")

        # --- 5. Load Data ---
        if processed_faqs:
            # Clear existing FAQs or update based on question_id
            # For simplicity, let's replace all for now. In production, consider upsert for updates.
            faqs_collection.delete_many({}) # Dangerous in production without careful planning
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
