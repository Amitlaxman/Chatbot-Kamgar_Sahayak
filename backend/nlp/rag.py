from typing import List, Dict, Any
import os
import json
import logging
from backend.nlp.similarity import get_embedding, cosine_similarity
from backend.db.mongo_utils import get_legal_db 

def retrieve_relevant_faqs(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """
    Retrieves the most relevant documents from all content collections in MongoDB.
    """
    logger = logging.getLogger(__name__)
    try:
        db = get_legal_db() 
        
        # 1. Get all collection names and filter out system/non-content collections.
        all_db_collections = db.list_collection_names()
        
        # Define all system and utility collections to exclude
        exclude_collections = {
            'logs', 'users', 'admin_marking', 'admin_answers', 'keywords',
            'fs.files', 'fs.chunks', 'admin_users', 'system.views',
            'system.version', 'system.profile'
        }
        
        # Identify relevant collections based on query content
        query_lower = query.lower()
        query_terms = set(query_lower.split())
        
        def get_collection_relevance(collection_name: str) -> float:
            name_lower = collection_name.lower()
            # Split on both underscores and spaces for better matching
            name_terms = set(name_lower.replace('_', ' ').split())
            # Calculate term overlap between query and collection name
            matching_terms = query_terms.intersection(name_terms)
            return len(matching_terms) / len(query_terms) if query_terms else 0
        
        # Filter content collections
        content_collections = [
            name for name in all_db_collections 
            if (not name.startswith('system.') 
                and not name.startswith('_') 
                and name not in exclude_collections)
        ]
        
        # Sort collections by relevance to query
        content_collections.sort(key=get_collection_relevance, reverse=True)
        
        logger.debug(f"All collections in DB: {all_db_collections}")
        logger.debug(f"Excluded collections: {exclude_collections}")
        logger.info(f"Content collections found (sorted by relevance): {content_collections}")
        
        if not content_collections:
            logger.warning("No content collections found in the database to search.")
            return []

        logger.info("Searching in collections: %s", ", ".join(content_collections))

        # 2. Fetch all chunks from all content collections.
        all_chunks = []
        for collection_name in content_collections:
            collection = db[collection_name]
            chunks = list(collection.find({}, {"_id": 1, "content_en": 1, "content_hi": 1, "embedding_en": 1, "embedding_hi": 1, "source": 1}))
            for chunk in chunks:
                chunk["_collection"] = collection_name  # Tag chunk with its source collection
            all_chunks.extend(chunks)
        
        if not all_chunks:
            logger.warning("No documents (chunks) found in any of the content collections.")
            return []

        # 3. Determine query language and select appropriate embedding field.
        import re
        is_hindi = re.search(r"[\u0900-\u097F]", query)
        emb_field = "embedding_hi" if is_hindi else "embedding_en"
        content_field = "content_hi" if is_hindi else "content_en"
        logger.info(f"Query language detected as {'Hindi' if is_hindi else 'English'}. Using '{emb_field}'.")

        # 4. Get embedding for the user's query.
        query_emb = get_embedding(query)

        # 5. Score each chunk based on cosine similarity.
        scored_chunks = []
        for chunk in all_chunks:
            chunk_emb = chunk.get(emb_field)
            
            # If embedding is missing, generate and store it on-the-fly.
            if not chunk_emb:
                content = chunk.get(content_field) or chunk.get("content_en") or ""
                if content:
                    try:
                        logger.warning(f"Embedding missing for chunk from '{chunk.get('source')}'. Generating on-the-fly.")
                        chunk_emb = get_embedding(content)
                        db[chunk["_collection"]].update_one(
                            {"_id": chunk["_id"]},
                            {"$set": {emb_field: chunk_emb.tolist()}}
                        )
                    except Exception as e:
                        logger.error(f"Could not generate on-the-fly embedding for chunk {chunk.get('_id')}: {e}")
                        continue
                else:
                    continue

            try:
                sim = cosine_similarity(query_emb, chunk_emb)
                scored_chunks.append((sim, chunk))
            except Exception as e:
                logger.warning(f"Could not calculate similarity for chunk {chunk.get('_id')}: {e}")

        if not scored_chunks:
            logger.info("No chunks were scored successfully.")
            return []

        # 6. Sort chunks by combined score (similarity and collection relevance)
        for sim, chunk in scored_chunks:
            # Combine semantic similarity with collection relevance
            combined_score = (sim * 0.7) + (chunk.get("_collection_relevance", 0) * 0.3)
            chunk["_combined_score"] = combined_score
        
        # Sort by combined score
        scored_chunks.sort(reverse=True, key=lambda x: x[1].get("_combined_score", x[0]))
        
        top_results = []
        for sim, chunk in scored_chunks[:top_k]:
            result = {
                "score": chunk.get("_combined_score", sim),
                "semantic_score": sim,
                "collection_relevance": chunk.get("_collection_relevance", 0),
                "source": chunk.get("source"),
                "content_en": chunk.get("content_en"),
                "content_hi": chunk.get("content_hi"),
                "_collection": chunk.get("_collection")
            }
            top_results.append(result)

        logger.info("Top %d matches: %s", len(top_results), 
                   "; ".join(f"Score: {res['score']:.3f} (Sem: {res['semantic_score']:.3f}, Rel: {res['collection_relevance']:.3f}) -> {res['source']}" 
                            for res in top_results))
        
        # --- Strong phrase/keyword match for legal action ---
        query_lower = query.lower()

        # --- Stricter phrase/section match ---
        import re
        # Extract all section numbers from the query
        section_numbers = re.findall(r'section\s*\d+[a-zA-Z()]*', query_lower)
        # Extract main legal action words (e.g., 'service separation', 'closure', etc.)
        legal_phrases = [
            # Employment/Separation
            'service separation', 'separation', 'termination', 'dismissal', 'removal', 'resignation',
            'retirement', 'superannuation', 'voluntary retirement', 'compulsory retirement',
            # Dispute/Claim Types
            'closure', 'retrenchment', 'layoff', 'wage claim', 'gratuity', 'bonus', 'overtime',
            'appeal', 'recovery', 'charter of demands', 'standing order', 'maternity', 'journalist',
            'factory', 'application', 'dispute', 'conciliation', 'prosecution', 'time limit', 'timeline',
            'non-payment', 'case disposal', 'industrial dispute', 'settlement', 'conciliation', 'adjudication',
            'award', 'reference', 'reinstatement', 'compensation', 'back wages', 'promotion', 'transfer',
            'suspension', 'show cause', 'charge sheet', 'enquiry', 'disciplinary', 'punishment',
            # Applicability/Scope
            'applicability', 'scope', 'applies to', 'extends to', 'whole of india', 'jurisdiction', 'coverage', 'extent', 'territorial', 'who does this act apply to', 'where does this act apply', 'pan india', 'state of', 'all of india', 'entire country',
            # Acts/Sections
            'industrial disputes act', 'minimum wages act', 'payment of wages act', 'equal remuneration act',
            'maternity benefit act', 'mp industrial relations act', 'working journalists act',
            'payment of gratuity act', 'beedi and cigar workers act', 'shops and establishments act',
            'standing orders act', 'factories act', 'labour law', 'labour department',
            # Misc
            'case', 'petition', 'application', 'order', 'notice', 'hearing', 'compliance', 'delay',
            'extension', 'limitation', 'period', 'duration', 'within', 'prescribed', 'specified', 'fixed',
            'mandate', 'requirement', 'obligation', 'responsibility', 'accountability', 'disciplinary action',
            'officer', 'employee', 'employer', 'worker', 'staff', 'management', 'union', 'committee',
            'board', 'tribunal', 'court', 'authority', 'government', 'department', 'ministry',
            # Hindi/vernacular (for bilingual queries)
            'सेवा पृथक्करण', 'समाप्ति', 'सेवानिवृत्ति', 'निलंबन', 'छंटनी', 'समझौता', 'मामला', 'आवेदन',
            'निर्णय', 'अधिनियम', 'धारा', 'समय सीमा', 'समय-सीमा', 'विलंब', 'कार्रवाई', 'शिकायत', 'अपील',
            'श्रम', 'कर्मचारी', 'नियोक्ता', 'प्रबंधन', 'संघ', 'बोर्ड', 'अधिकार', 'प्राधिकरण', 'सरकार',
        ]
        # --- Applicability/Scope post-filter ---
        applicability_words = ['applicability', 'scope', 'applies to', 'extends to', 'whole of india', 'jurisdiction', 'coverage', 'extent', 'who does this act apply to', 'where does this act apply', 'pan india', 'all of india', 'entire country', 'state of']
        if any(word in query_lower for word in applicability_words):
            applicability_chunks = []
            for sim, chunk in scored_chunks:
                for field in ["source", "content_en", "content_hi"]:
                    val = (chunk.get(field) or "").lower()
                    if any(word in val for word in applicability_words):
                        applicability_chunks.append((sim + 1.0, chunk))
                        break
            if applicability_chunks:
                applicability_chunks = list({id(c[1]): c for c in applicability_chunks}.values())
                applicability_chunks.sort(reverse=True, key=lambda x: x[0])
                return [chunk for sim, chunk in applicability_chunks[:top_k]]
        matched_phrases = [phrase for phrase in legal_phrases if phrase in query_lower]
        strong_chunks = []
        for sim, chunk in scored_chunks:
            for field in ["source", "content_en", "content_hi"]:
                val = (chunk.get(field) or "").lower()
                # Require all matched phrases to be present in the chunk
                if matched_phrases and all(phrase in val for phrase in matched_phrases):
                    # If section number is in query, prefer chunk with same section
                    if section_numbers:
                        if any(sec in val for sec in section_numbers):
                            strong_chunks.append((sim + 1.0, chunk))  # boost score for perfect match
                        else:
                            continue
                    else:
                        strong_chunks.append((sim, chunk))
        if strong_chunks:
            strong_chunks = list({id(c[1]): c for c in strong_chunks}.values())
            strong_chunks.sort(reverse=True, key=lambda x: x[0])
            return [chunk for sim, chunk in strong_chunks[:top_k]]

        # --- Fallback: Post-filter for section/word match as before ---
        preferred_chunks = []
        for sim, chunk in scored_chunks:
            for field in ["source", "content_en", "content_hi"]:
                val = (chunk.get(field) or "").lower()
                import re
                section_match = re.search(r'section\s*\d+[a-zA-Z()]*', query_lower)
                if section_match and section_match.group(0) in val:
                    preferred_chunks.append((sim, chunk))
                    break
                for word in query_lower.split():
                    if len(word) > 3 and word in val:
                        preferred_chunks.append((sim, chunk))
                        break
        if preferred_chunks:
            preferred_chunks = list({id(c[1]): c for c in preferred_chunks}.values())
            preferred_chunks.sort(reverse=True, key=lambda x: x[0])
            return [chunk for sim, chunk in preferred_chunks[:top_k]]
        # Otherwise, return the full chunk data for the top results, sorted by combined score
        return [chunk for sim, chunk in scored_chunks[:top_k]]
    except Exception as e:
        logger.error(f"Error in retrieve_relevant_faqs: {e}", exc_info=True)
        return []

def format_context_for_generation(faqs: List[Dict[str, Any]], language: str = 'en') -> str:
    lines = []
    for i, chunk in enumerate(faqs, 1):
        # Try to get content in the requested language, fallback to other language if needed
        text = (
            chunk.get(f'content_{language}') or
            chunk.get('content_en' if language == 'hi' else 'content_hi') or
            chunk.get('content') or
            chunk.get('text') or
            chunk.get('source', '') or
            str(chunk)
        ).strip()
        
        # Try to get topic from collection name if not in chunk
        topic = chunk.get('topic', '')
        if not topic and '_collection' in chunk:
            topic = chunk['_collection'].replace('_', ' ').title()
        
        lines.append(f"Chunk {i} (Topic: {topic}):\n{text}\n")
    
    return "\n".join(lines)
