import json
from pymongo import MongoClient
import os

# Usage: python ingest_links.py <path_to_collection_reference_links.json>

MONGO_URI = os.getenv('MONGO_URI', 'mongodb+srv://Elixir:elixir@clutter.3ary0d9.mongodb.net/?retryWrites=true&w=majority')
DB_NAME = 'legal_db'  # Use the same DB as your main content
COLLECTION_NAME = 'links'  # New collection for reference links


def ingest_links(json_path):
    with open(json_path, encoding='utf-8') as f:
        data = json.load(f)
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    col = db[COLLECTION_NAME]
    col.delete_many({})  # Clear existing
    if isinstance(data, list):
        col.insert_many(data)
    else:
        col.insert_one(data)
    print(f"Ingested {len(data) if isinstance(data, list) else 1} records into {DB_NAME}.{COLLECTION_NAME}")

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Usage: python ingest_links.py <path_to_collection_reference_links.json>')
        exit(1)
    ingest_links(sys.argv[1])
