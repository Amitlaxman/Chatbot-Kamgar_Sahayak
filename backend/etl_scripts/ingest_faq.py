import pandas as pd
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from sentence_transformers import SentenceTransformer
import os
import logging
from dotenv import load_dotenv
from typing import List, Dict, Any

# --- Configuration ---
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME", "chatbot_db")
FAQ_COLLECTION = os.getenv("FAQ_COLLECTION", "faqs")
NLP_MODEL_NAME = os.getenv("NLP_MODEL_NAME", "paraphrase-multilingual-MiniLM-L12-v2")

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_etl(faq_data_path: str):
    """
    Runs the ETL pipeline to ingest FAQ data into MongoDB,
    generating embeddings for each FAQ.
    This version handles CSVs with multiple rows per FAQ (one per language).
    """
    logger.info("Starting ETL pipeline...")
    client = None
    model = None
    try:
        # --- 1. Connect to MongoDB ---
        logger.info("Connecting to MongoDB for ETL...")
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
        # Read the CSV, assuming 'ID', 'category', 'language', 'Question', 'Answer', 'keywords_en', 'keywords_hi' columns
        df = pd.read_csv(faq_data_path, keep_default_na=False) # Keep_default_na=False prevents empty strings from being read as NaN
        logger.info(f"Extracted {len(df)} rows from CSV.")

        # --- 4. Transform Data (Group, Generate Embeddings & Clean) ---
        # Group by unique FAQ ID
        grouped_faqs = df.groupby('ID')
        
        processed_faqs = []
        for faq_id, group in grouped_faqs:
            faq_doc: Dict[str, Any] = {
                "question_id": str(faq_id),
                "keywords_en": [],
                "keywords_hi": [],
                "answer_en": "",
                "answer_hi": "",
                "category": group['category'].iloc[0] if not group['category'].empty else 'General',
                "embedding": []
            }
            
            combined_text_for_embedding = []

            for _, row in group.iterrows():
                lang = str(row['language']).lower()
                question_text = str(row['Question']).strip()
                answer_text = str(row['Answer']).strip()
                
                # Combine question and answer for a more robust embedding
                text_to_embed_from_row = question_text + " " + answer_text
                if text_to_embed_from_row.strip():
                    combined_text_for_embedding.append(text_to_embed_from_row.strip())

                if lang == 'en':
                    faq_doc['answer_en'] = answer_text
                    if 'keywords_en' in row and row['keywords_en']:
                        faq_doc['keywords_en'] = [k.strip() for k in str(row['keywords_en']).split(',')]
                elif lang == 'hi':
                    faq_doc['answer_hi'] = answer_text
                    if 'keywords_hi' in row and row['keywords_hi']:
                        faq_doc['keywords_hi'] = [k.strip() for k in str(row['keywords_hi']).split(',')]
                elif lang == 'hinglish':
                    if not faq_doc['answer_hi']:
                         faq_doc['answer_hi'] = answer_text
                    if not faq_doc['keywords_hi'] and 'keywords_hi' in row and row['keywords_hi']:
                        faq_doc['keywords_hi'] = [k.strip() for k in str(row['keywords_hi']).split(',')]
            
            text_for_embedding = " ".join(combined_text_for_embedding).strip()
            if not text_for_embedding:
                logger.warning(f"Skipping FAQ ID {faq_id} due to missing text for embedding across all languages.")
                continue

            # Generate embedding
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
        logger.info("Creating a dummy 'faqs_raw.csv' for demonstration. Please populate it with your actual data.")
        
        # Creating a dummy CSV that matches the *new* expected input format
        dummy_data_rows = [
            {'ID': 1, 'category': 'Wages & Payments', 'language': 'en', 'Question': 'What is the minimum wage?', 'Answer': 'Min wage is X.', 'keywords_en': 'wage, salary', 'keywords_hi': 'मजदूरी, वेतन'},
            {'ID': 1, 'category': 'Wages & Payments', 'language': 'hi', 'Question': 'न्यूनतम मजदूरी क्या है?', 'Answer': 'न्यूनतम मजदूरी X है।', 'keywords_en': 'wage, salary', 'keywords_hi': 'मजदूरी, वेतन'},
            {'ID': 1, 'category': 'Wages & Payments', 'language': 'hinglish', 'Question': 'Minimum wage kitna hai?', 'Answer': 'Min wage X hai.', 'keywords_en': 'wage, salary', 'keywords_hi': 'मजदूरी, वेतन'},
            {'ID': 2, 'category': 'Health & Safety', 'language': 'en', 'Question': 'Safety gear?', 'Answer': 'Helmet, gloves, shoes.', 'keywords_en': 'safety, gear', 'keywords_hi': 'सुरक्षा, उपकरण'},
            {'ID': 2, 'category': 'Health & Safety', 'language': 'hi', 'Question': 'सुरक्षा उपकरण?', 'Answer': 'हेलमेट, दस्ताने, जूते।', 'keywords_en': 'safety, gear', 'keywords_hi': 'सुरक्षा, उपकरण'},
        ]
        pd.DataFrame(dummy_data_rows).to_csv(faq_data_file, index=False)
        
    run_etl(faq_data_file)