import os
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv(dotenv_path=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.env')))
MONGO_URI = os.getenv('MONGO_URI')
DB_NAME = os.getenv('DB_NAME', 'legal_db')

if not MONGO_URI:
    print('MONGO_URI not found in .env')
    raise SystemExit(1)

client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
db = client[DB_NAME]

print(f"Connected to DB: {DB_NAME}")
collections = [c for c in db.list_collection_names() if not c.startswith('system.')]
print(f"Collections ({len(collections)}): {collections}")

for cname in collections:
    coll = db[cname]
    total = coll.count_documents({})
    emb_en = coll.count_documents({"embedding_en": {"$exists": True}})
    emb_hi = coll.count_documents({"embedding_hi": {"$exists": True}})
    print(f"- {cname}: total={total}, embedding_en={emb_en}, embedding_hi={emb_hi}")

# Show sample document for a target collection if exists
sample_col = None
if collections:
    sample_col = collections[0]

if sample_col:
    doc = db[sample_col].find_one({}, {'_id':0})
    print('\nSample doc from', sample_col)
    print(doc)

client.close()
