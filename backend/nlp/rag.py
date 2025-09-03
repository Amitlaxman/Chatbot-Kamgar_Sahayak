
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
        if not collection_names:
            logger.warning("No collections found in database.")
            return []
        
        logger.info(f"Searching collections: {', '.join(collection_names)}")
        all_chunks = []
        chunks_by_collection = {}
        for cname in collection_names:
            collection = db[cname]
            chunks = list(collection.find({}, {"_id": 0}))
            chunks_by_collection[cname] = len(chunks)
            all_chunks.extend(chunks)
        logger.info(f"Collection sizes: {', '.join(f'{c}: {n}' for c,n in chunks_by_collection.items())}")
        
        if not all_chunks:
            logger.warning("No chunks found in any collection.")
            return []

        # Detect language: if query has Devanagari, use Hindi embedding, else English
        import re
        if re.search(r'[\u0900-\u097F]', query):
            emb_field = 'embedding_hi'
            content_field = 'content_hi'
            logger.info("Detected Hindi query, using Hindi embeddings")
        else:
            emb_field = 'embedding_en'
            content_field = 'content_en'
            logger.info("Using English embeddings")

        query_emb = get_embedding(query)
        logger.info(f"Query embedding length: {len(query_emb) if hasattr(query_emb,'__len__') else 'unknown'}; using emb_field='{emb_field}'")

        scored = []
        missing_emb_count = 0
        has_content_count = 0
        for idx, chunk in enumerate(all_chunks):
            if chunk.get(content_field):
                has_content_count += 1
            
            chunk_emb = chunk.get(emb_field)
            if not chunk_emb:
                missing_emb_count += 1
                text = chunk.get(content_field, '') or chunk.get('source', '')
                if text:
                    try:
                        chunk_emb = get_embedding(text)  # compute on-the-fly
                    except Exception as e:
                        logger.warning(f"Failed to compute embedding for chunk {idx} ({chunk.get('source')}): {e}")
                        continue
                else:
                    logger.debug(f"Skipping chunk {idx} - no content to embed")
                    continue
            
            try:
                sim = cosine_similarity(query_emb, chunk_emb)
                scored.append((sim, chunk))
            except Exception as e:
                logger.warning(f"Similarity failed for chunk {idx} ({chunk.get('source')}): {e}")

        logger.info(f"Processing stats: {len(all_chunks)} total chunks, {has_content_count} have {content_field}, {missing_emb_count} missing {emb_field}")

        if not scored:
            logger.info("No chunks scored successfully.")
            return []

        scored.sort(reverse=True, key=lambda x: x[0])
        top_scores = [(float(s), c.get('source', '')) for s, c in scored[:top_k]]
        logger.info(f"Top {len(top_scores)} matches: " + '; '.join(f"{score:.3f} -> {source}" for score,source in top_scores))
        
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
