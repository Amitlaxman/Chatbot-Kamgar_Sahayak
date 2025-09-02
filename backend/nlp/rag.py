
from typing import List, Dict, Any
import pymongo
import os
import json
from backend.nlp.similarity import get_embedding, cosine_similarity


from pymongo import MongoClient

def retrieve_relevant_faqs(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'Minimum_Wages_Act_1948.json'))
    with open(data_path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    if not chunks:
        return []
    query_emb = get_embedding(query)
    scored = []
    for chunk in chunks:
        # Combining English and Hindi content for better matching
        text = (chunk.get('content_en', '') or '') + ' ' + (chunk.get('content_hi', '') or '')
        emb = get_embedding(text)
        sim = cosine_similarity(query_emb, emb)
        scored.append((sim, chunk))
    scored.sort(reverse=True, key=lambda x: x[0])
    # Connect to MongoDB Atlas and fetch chunks from the relevant collection
    MONGO_URI = os.getenv("MONGO_URI")
    DB_NAME = os.getenv("DB_NAME", "legal_db")
    COLLECTION_NAME = os.getenv("CHUNKS_COLLECTION", "Minimum_Wages_Act_1948")
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]
    chunks = list(collection.find({}, {"_id": 0}))
    if not chunks:
        return []
    query_emb = get_embedding(query)
    scored = []
    for chunk in chunks:
        text = (chunk.get('content_en', '') or '') + ' ' + (chunk.get('content_hi', '') or '')
        emb = get_embedding(text)
        sim = cosine_similarity(query_emb, emb)
        scored.append((sim, chunk))
    scored.sort(reverse=True, key=lambda x: x[0])
    client.close()
    return [chunk for sim, chunk in scored[:top_k]]


def format_context_for_generation(faqs: List[Dict[str, Any]], language: str = 'en') -> str:

    lines = []
    for i, chunk in enumerate(faqs, 1):
        text = chunk.get('content_hi', '') if language == 'hi' else chunk.get('content_en', '')
        lines.append(f"Chunk {i}:\n{text}\n")
    return '\n'.join(lines)
