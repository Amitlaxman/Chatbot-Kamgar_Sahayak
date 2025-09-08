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
LINKS_COLLECTION_NAME = "links" 
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

def get_links_map(db_client):
    """Fetches all links and maps them by their source collection name."""
    db = db_client[DB_NAME]
    links_collection = db[LINKS_COLLECTION_NAME]
    links_map = {}
    for link_doc in links_collection.find():
        collection_name = link_doc.get("collection_name")
        link_url = link_doc.get("link")
        if collection_name and link_url:
            links_map[collection_name] = link_url
    print(f"🔗 Loaded {len(links_map)} links from '{LINKS_COLLECTION_NAME}' collection.")
    return links_map

def load_and_chunk_documents(db_client, text_splitter, links_map):
    """
    Loads documents from all collections, cleans content, adds source links,
    and splits them into chunks.
    """
    db = db_client[DB_NAME]
    all_chunks = []

    collections_to_process = [
        name for name in db.list_collection_names()
        if not name.startswith("system.") and name not in [VECTOR_COLLECTION_NAME, LINKS_COLLECTION_NAME]
    ]
    print(f"Discovered {len(collections_to_process)} content collections to process.")

    for coll_name in tqdm(collections_to_process, desc="Processing Collections"):
        collection = db[coll_name]
        
        for doc in collection.find():
            content_en = doc.get("content_en", "")
            content_hi = doc.get("content_hi", "")

            if not isinstance(content_en, str): content_en = ""
            if not isinstance(content_hi, str): content_hi = ""

            combined_text = []
            if content_en.strip():
                combined_text.append(f"English Content:\n{content_en.strip()}")
            if content_hi.strip():
                combined_text.append(f"Hindi Content:\n{content_hi.strip()}")
            
            final_text = "\n\n---\n\n".join(combined_text)

            if final_text:
                source_link = links_map.get(coll_name)
                # Add a log to show when a link is found
                if source_link:
                    print(f"  -> Found and attached link for '{coll_name}'")

                metadata = {
                    "source_collection": coll_name,
                    "original_id": str(doc.get("_id", "")),
                    "source_link": source_link
                }
                
                langchain_doc = Document(page_content=final_text, metadata=metadata)
                chunks = text_splitter.split_documents([langchain_doc])
                all_chunks.extend(chunks)

    return all_chunks


def main():
    """Main function to run the indexing process."""
    print("--- Starting the indexing process with link enrichment ---")

    try:
        client = MongoClient(MONGO_URI)
        client.admin.command('ping')
        print("✅ MongoDB connection successful.")
    except Exception as e:
        print(f"❌ Could not connect to MongoDB: {e}")
        return

    links_map = get_links_map(client)
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)

    print("📚 Loading and chunking documents...")
    all_document_chunks = load_and_chunk_documents(client, text_splitter, links_map)
    if not all_document_chunks:
        print("⚠️ No documents found to index. Exiting.")
        return
    print(f"✅ Created {len(all_document_chunks)} document chunks.")

    print("🧠 Initializing embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    print("✅ Embedding model loaded.")

    db = client[DB_NAME]
    vector_collection = db[VECTOR_COLLECTION_NAME]

    print(f"🗑️ Deleting all existing documents from '{VECTOR_COLLECTION_NAME}'...")
    delete_result = vector_collection.delete_many({})
    print(f"✅ Deleted {delete_result.deleted_count} documents.")

    print(f"🚀 Adding {len(all_document_chunks)} chunks to Atlas Vector Search...")
    MongoDBAtlasVectorSearch.from_documents(
        documents=all_document_chunks,
        embedding=embeddings,
        collection=vector_collection,
        index_name="vector_index",
    )
    print("✅ All documents have been successfully indexed.")
    print("--- Indexing process complete! ---")
    client.close()

if __name__ == "__main__":
    main()

