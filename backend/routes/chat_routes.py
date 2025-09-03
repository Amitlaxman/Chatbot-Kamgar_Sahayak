from fastapi import APIRouter, HTTPException, status
from backend.models.chat_model import ChatQuery, ChatResponse, LogEntry
from pydantic import BaseModel, Field
from backend.db.mongo_utils import get_mongo_db, insert_log_entry, get_all_faqs
from backend.nlp.similarity import get_embedding, cosine_similarity
from backend.nlp.rag import retrieve_relevant_faqs, format_context_for_generation
import logging
from datetime import datetime
from typing import Dict, Any, List
import os

logger = logging.getLogger(__name__)
router = APIRouter()

# Configuration
CONFIDENCE_THRESHOLD = 0.1  # lowered threshold to increase recall
KEYWORDS_COLLECTION = os.getenv("KEYWORDS_COLLECTION", "keywords")


async def get_synonyms_from_db(query_text: str, language: str) -> List[str]:
    """
    Retrieve synonyms from MongoDB
    """
    db = get_mongo_db()
    synonyms_collection = db[KEYWORDS_COLLECTION]

    query_words = [word.strip().lower() for word in query_text.split() if word.strip()]

    search_field = "english_synonyms" if language == 'en' else "hindi_synonyms"

    synonym_docs = list(synonyms_collection.find({search_field: {"$in": query_words}}))

    expanded_keywords = []
    for doc in synonym_docs:
        expanded_keywords.append(doc.get('english_keyword', ''))
        expanded_keywords.append(doc.get('hindi_keyword', ''))
        expanded_keywords.extend(doc.get('english_synonyms', []))
        expanded_keywords.extend(doc.get('hindi_synonyms', []))

    # Remove duplicates and empty strings
    expanded_keywords = [kw for kw in set(expanded_keywords) if kw]

    return expanded_keywords


from typing import Optional, List, Dict

class ChatRequest(BaseModel):
    user_id: str = Field(...)
    query_text: str = Field(...)
    language: str = Field("en")
    chat_history: Optional[List[Dict[str, str]]] = None

@router.post("/chat", response_model=ChatResponse)
async def chat_with_bot(request: ChatRequest):
    db = get_mongo_db()

    user_query_text = request.query_text
    user_id = request.user_id
    language = request.language
    chat_history = request.chat_history

    logger.info(f"Received chat query from {user_id} ({language}): '{user_query_text}'")

    # --- Chat history support ---
    context_window = 5
    history_context = ""
    if chat_history:
        for msg in chat_history[-context_window:]:
            prefix = "User:" if msg.get("sender") == "user" else "Bot:"
            history_context += f"{prefix} {msg.get('text', '')}\n"

    bot_response_text = ""
    status_text = "unanswered"
    similarity_score = None

    try:
        # 1. Expand query with synonyms
        synonym_keywords = await get_synonyms_from_db(user_query_text, language)
        logger.info(f"Synonyms for query '{user_query_text}': {synonym_keywords}")

        expanded_query_text = user_query_text + " " + " ".join(synonym_keywords)
        logger.info(f"Expanded query text for embedding: '{expanded_query_text}'")

        # --- RAG: Retrieve top relevant chunks as context, using chat history ---
        rag_query = (history_context + "User: " + expanded_query_text).strip()
        top_chunks = retrieve_relevant_faqs(rag_query, top_k=3)
        context_str = format_context_for_generation(top_chunks, language=language)
        logger.info(f"RAG context for generation:\n{context_str}")

        # Compute similarity scores for debugging using stored embeddings
        if top_chunks:
            query_emb = get_embedding(rag_query)
            scored_chunks = []
            import re
            use_hi = bool(re.search(r'[\u0900-\u097F]', rag_query))
            emb_field = 'embedding_hi' if use_hi else 'embedding_en'
            for idx, chunk in enumerate(top_chunks):
                chunk_emb = chunk.get(emb_field) or chunk.get('embedding') or chunk.get('embed')
                if not chunk_emb:
                    # fallback to generate embedding from available text
                    text = (chunk.get('content_hi') if use_hi else chunk.get('content_en')) or chunk.get('source', '')
                    try:
                        chunk_emb = get_embedding(text)
                    except Exception as e:
                        logger.warning(f"Failed to generate fallback embedding for chunk idx={idx}: {e}")
                        continue
                sim = cosine_similarity(query_emb, chunk_emb)
                scored_chunks.append((sim, chunk))
            if not scored_chunks:
                logger.info("No scored chunks after attempting to use stored embeddings and fallbacks.")
                best_score, best_chunk = (0.0, None)
                similarity_score = None
            else:
                scored_chunks.sort(reverse=True, key=lambda x: x[0])
                logger.info(f"Similarity scores for top chunks: {[round(float(s),3) for s,_ in scored_chunks]}")
                best_score, best_chunk = scored_chunks[0]
                similarity_score = float(best_score)
                logger.info(f"Selected chunk with score {similarity_score}")
            # Always return the top chunk, even if below threshold
            if best_chunk:
                if language == 'hi':
                    bot_response_text = best_chunk.get('content_hi') or best_chunk.get('content_en') or best_chunk.get('source', 'उत्तर उपलब्ध नहीं है।')
                else:
                    bot_response_text = best_chunk.get('content_en') or best_chunk.get('content_hi') or best_chunk.get('source', 'Answer not available.')
            else:
                bot_response_text = ('I am sorry, I could not find a relevant chunk. Your query has been logged for review.' if language != 'hi' 
                                     else 'माफ़ कीजिए, मैं उपयुक्त जानकारी नहीं ढूँढ पाया। आपका प्रश्न समीक्षा के लिए लॉग कर दिया गया है।')
            status_text = "answered" if similarity_score >= CONFIDENCE_THRESHOLD else "low_confidence"
        else:
            bot_response_text = ("I'm sorry, I don't have a precise answer for that right now. "
                                 "Your query has been noted for review by our team.")
            status_text = "unanswered"
            try:
                from backend.services.email_service import send_email
                subject = "Unanswered Query Notification"
                body = (
                    f"A new unanswered query was received.\n\n"
                    f"User ID: {user_id}\n"
                    f"Query Text: {user_query_text}\n"
                    f"Language: {language}\n"
                    f"Timestamp: {datetime.now()}\n"
                    f"Context: {context_str}\n"
                )
                admin_email = os.getenv("ADMIN_EMAIL_RECEIVER", "kaaamgar.sahayak@gmail.com")
                send_email(subject, body, admin_email)
                logger.info(f"Unanswered query email sent to {admin_email}")
            except Exception as e:
                logger.error(f"Failed to send unanswered query email: {e}", exc_info=True)

    except Exception as e:
        import traceback
        logger.error(f"Error processing chat query '{user_query_text}': {e}\n{traceback.format_exc()}")
        bot_response_text = "An internal error occurred while processing your request. Please try again."
        status_text = "error"

        await insert_log_entry(LogEntry(
            timestamp=datetime.now(),
            user_id=user_id,
            query_text=user_query_text,
            bot_response_text=bot_response_text,
            status=status_text,
            language=language,
            similarity_score=similarity_score
        ).dict())
        raise HTTPException(status_code=500, detail="Internal server error.")

    if status_text != "error":
        await insert_log_entry(LogEntry(
            timestamp=datetime.now(),
            user_id=user_id,
            query_text=user_query_text,
            bot_response_text=bot_response_text,
            status=status_text,
            language=language,
            similarity_score=similarity_score
        ).dict())

    return ChatResponse(
        bot_response=bot_response_text,
        status=status_text,
        language=language,
        similarity_score=similarity_score
    )
