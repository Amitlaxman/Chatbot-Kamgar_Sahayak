
from typing import List, Dict, Any
import os
import json
import logging
from backend.nlp.similarity import get_embedding, cosine_similarity
from backend.db.mongo_utils import get_mongo_db



def retrieve_relevant_faqs(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    logger = logging.getLogger(__name__)
    try:
        db = get_mongo_db()
        # Search all collections except system collections
        collection_names = [c for c in db.list_collection_names() if not c.startswith('system.')]
        all_chunks = []
        for cname in collection_names:
            collection = db[cname]
            chunks = list(collection.find({}, {"_id": 0}))
            all_chunks.extend(chunks)
        if not all_chunks:
            logger.warning("No chunks found in any collection.")
            return []
        query_emb = get_embedding(query)
        scored = []
        for chunk in all_chunks:
            text = (chunk.get('content_en', '') or '') + ' ' + (chunk.get('content_hi', '') or '')
            emb = get_embedding(text)
            sim = cosine_similarity(query_emb, emb)
            scored.append((sim, chunk))
        scored.sort(reverse=True, key=lambda x: x[0])
        return [chunk for sim, chunk in scored[:top_k]]
    except Exception as e:
        logger.error(f"Error in retrieve_relevant_faqs: {e}", exc_info=True)
        return []


def format_context_for_generation(faqs: List[Dict[str, Any]], language: str = 'en') -> str:

    lines = []
    for i, chunk in enumerate(faqs, 1):
        text = chunk.get('content_hi', '') if language == 'hi' else chunk.get('content_en', '')
        lines.append(f"Chunk {i}:\n{text}\n")
    return '\n'.join(lines)
