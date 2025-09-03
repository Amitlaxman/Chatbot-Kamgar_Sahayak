
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
        # Detect language: if query has Devanagari, use Hindi embedding, else English
        import re
        if re.search(r'[\u0900-\u097F]', query):
            emb_field = 'embedding_hi'
        else:
            emb_field = 'embedding_en'

        query_emb = get_embedding(query)
        logger.info(f"Query embedding length: {len(query_emb) if hasattr(query_emb,'__len__') else 'unknown'}; using emb_field='{emb_field}'")

        scored = []
        missing_emb_count = 0
        for idx, chunk in enumerate(all_chunks):
            chunk_emb = chunk.get(emb_field)
            if not chunk_emb:
                # try common alternate field names
                chunk_emb = chunk.get('embedding') or chunk.get('embed')
            if chunk_emb:
                try:
                    sim = cosine_similarity(query_emb, chunk_emb)
                    scored.append((sim, chunk))
                except Exception as e:
                    logger.warning(f"Failed similarity calc for chunk idx={idx} topic={chunk.get('source')} : {e}")
            else:
                # fallback: compute embedding on-the-fly from available text
                missing_emb_count += 1
                text = (chunk.get('content_en') or '') if emb_field == 'embedding_en' else (chunk.get('content_hi') or '')
                if not text:
                    # fallback to source
                    text = chunk.get('source', '')
                if text:
                    try:
                        temp_emb = get_embedding(text)
                        sim = cosine_similarity(query_emb, temp_emb)
                        scored.append((sim, chunk))
                    except Exception as e:
                        logger.warning(f"On-the-fly embedding failed for chunk idx={idx}: {e}")
                else:
                    logger.debug(f"Skipping chunk idx={idx} because no content available to compute embedding.")

        if missing_emb_count:
            logger.info(f"Chunks missing precomputed embedding: {missing_emb_count}. Used on-the-fly embeddings for those.")

        if not scored:
            logger.info("No scored chunks after similarity checks.")
            return []

        scored.sort(reverse=True, key=lambda x: x[0])
        logging.info(f"Similarity scores for top chunks: {[float(s) for s,_ in scored[:top_k]]}")
        return [chunk for sim, chunk in scored[:top_k]]
    except Exception as e:
        logger.error(f"Error in retrieve_relevant_faqs: {e}", exc_info=True)
        return []


def format_context_for_generation(faqs: List[Dict[str, Any]], language: str = 'en') -> str:

    lines = []
    for i, chunk in enumerate(faqs, 1):
        text = chunk.get('content_hi', '') if language == 'hi' else chunk.get('content_en', '')
        if not text:
            text = chunk.get('source', '') or str(chunk)
        lines.append(f"Chunk {i} (Topic: {chunk.get('topic','')}):\n{text}\n")
    return '\n'.join(lines)
