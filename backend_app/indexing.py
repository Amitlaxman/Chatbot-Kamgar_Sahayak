import os
from dotenv import load_dotenv
from pymongo import MongoClient
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from tqdm import tqdm

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = "legal_db"
VECTOR_COLLECTION_NAME = "legal_vectors"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2" # Must match the model in main.py

def load_and_chunk_documents(db_client, text_splitter):
    """
    Loads documents from all collections in the database (except the vector one),
    cleans the content, and splits them into chunks.
    """
    db = db_client[DB_NAME]
    all_chunks = []

    # Get all collection names dynamically, excluding system and vector collections
    collections_to_process = [
        name for name in db.list_collection_names()
        if not name.startswith("system.") and name != VECTOR_COLLECTION_NAME
    ]
    print(f"Discovered {len(collections_to_process)} collections to process.")

    for coll_name in tqdm(collections_to_process, desc="Processing Collections"):
        collection = db[coll_name]
        
        for doc in collection.find():
            # --- THIS IS THE CRITICAL FIX ---
            # Safely get 'content_en' and 'content_hi', ensuring they are strings
            content_en = doc.get("content_en", "")
            content_hi = doc.get("content_hi", "")

            if not isinstance(content_en, str):
                content_en = ""
            if not isinstance(content_hi, str):
                content_hi = ""

            # Combine content into a single, clean text field
            combined_text = []
            if content_en.strip():
                combined_text.append(f"English Content:\n{content_en.strip()}")
            if content_hi.strip():
                combined_text.append(f"Hindi Content:\n{content_hi.strip()}")
            
            final_text = "\n\n---\n\n".join(combined_text)
            # --- END OF CRITICAL FIX ---

            if final_text:
                # Create a LangChain Document with clean page_content
                langchain_doc = Document(
                    page_content=final_text,
                    metadata={
                        "source_collection": coll_name,
                        "original_id": str(doc.get("_id", "")),
                    },
                )
                # Split the clean document into smaller chunks
                chunks = text_splitter.split_documents([langchain_doc])
                all_chunks.extend(chunks)

    return all_chunks


def main():
    """Main function to run the indexing process."""
    print("--- Starting the indexing process ---")

    # 1. Initialize MongoDB Client
    try:
        client = MongoClient(MONGO_URI)
        client.admin.command('ping')
        print("✅ MongoDB connection successful.")
    except Exception as e:
        print(f"❌ Could not connect to MongoDB: {e}")
        return

    # 2. Define Text Splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)

    # 3. Load and Chunk Documents
    print("📚 Loading and chunking documents from all collections...")
    all_document_chunks = load_and_chunk_documents(client, text_splitter)
    if not all_document_chunks:
        print("⚠️ No documents found to index. Exiting.")
        return
    print(f"✅ Created {len(all_document_chunks)} document chunks.")

    # 4. Initialize Embeddings Model
    print("🧠 Initializing embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    print("✅ Embedding model loaded.")

    # 5. Get reference to the target collection for vectors
    db = client[DB_NAME]
    vector_collection = db[VECTOR_COLLECTION_NAME]

    # 6. Delete existing documents to ensure a clean slate
    print(f"🗑️ Deleting all existing documents from '{VECTOR_COLLECTION_NAME}'...")
    delete_result = vector_collection.delete_many({})
    print(f"✅ Deleted {delete_result.deleted_count} documents.")

    # 7. Add documents to MongoDB Atlas Vector Search
    print(f"🚀 Adding {len(all_document_chunks)} chunks to Atlas Vector Search...")
    MongoDBAtlasVectorSearch.from_documents(
        documents=all_document_chunks,
        embedding=embeddings,
        collection=vector_collection,
        index_name="vector_index",  # Make sure this matches your index name in Atlas
    )
    print("✅ All documents have been successfully indexed.")
    print("--- Indexing process complete! ---")
    client.close()

if __name__ == "__main__":
    main()

