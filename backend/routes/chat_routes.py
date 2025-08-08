from fastapi import APIRouter, HTTPException, status
from backend.models.chat_model import ChatQuery, ChatResponse, LogEntry
from backend.db.mongo_utils import get_mongo_db, insert_log_entry, get_all_faqs
from backend.nlp.similarity import get_embedding, cosine_similarity
import logging
from datetime import datetime
from typing import Dict, Any

logger = logging.getLogger(__name__)
router = APIRouter()

# --- Configuration for Chatbot Logic ---
# You can adjust this confidence threshold based on testing
CONFIDENCE_THRESHOLD = 0.75 # If similarity is below this, query is "unanswered"

@router.post("/chat", response_model=ChatResponse)
async def chat_with_bot(query: ChatQuery):
    """
    Processes a user's chat query, generates embeddings, finds the best FAQ match,
    logs the interaction, and returns the bot's response.
    """
    db = get_mongo_db() # Ensure DB connection is active
    
    user_query_text = query.query_text
    user_id = query.user_id
    language = query.language

    logger.info(f"Received chat query from {user_id} ({language}): '{user_query_text}'")

    bot_response_text = ""
    status_text = "unanswered"
    similarity_score = None
    
    try:
        # 1. Generate Embedding for User Query
        user_embedding = get_embedding(user_query_text)

        # 2. Retrieve all FAQs with their embeddings
        all_faqs = await get_all_faqs()
        if not all_faqs:
            logger.warning("No FAQs found in the database. Returning default unanswered response.")
            bot_response_text = "I'm sorry, my knowledge base is currently empty. Please try again later."
            status_text = "unanswered"
            # Log and return
            await insert_log_entry(LogEntry(
                timestamp=datetime.now(),
                user_id=user_id,
                query_text=user_query_text,
                bot_response_text=bot_response_text,
                status=status_text,
                language=language
            ).dict())
            return ChatResponse(
                bot_response=bot_response_text,
                status=status_text,
                language=language
            )

        # 3. Find the Best Matching FAQ
        best_match_faq = None
        highest_similarity = -1.0

        for faq in all_faqs:
            if 'embedding' not in faq or not faq['embedding']:
                logger.warning(f"FAQ with ID {faq.get('question_id', 'N/A')} is missing embedding. Skipping.")
                continue
            
            faq_embedding = faq['embedding']
            current_similarity = cosine_similarity(user_embedding, faq_embedding)

            if current_similarity > highest_similarity:
                highest_similarity = current_similarity
                best_match_faq = faq

        similarity_score = highest_similarity
        logger.info(f"Highest similarity found: {highest_similarity:.4f} for FAQ ID: {best_match_faq.get('question_id', 'N/A') if best_match_faq else 'None'}")

        # 4. Determine Response based on Confidence Threshold
        if best_match_faq and highest_similarity >= CONFIDENCE_THRESHOLD:
            # Check for language-specific answer
            if language == 'hi' and 'answer_hi' in best_match_faq:
                bot_response_text = best_match_faq['answer_hi']
            elif language == 'hinglish' and 'answer_hi' in best_match_faq: # Assuming Hinglish uses Hindi answer
                bot_response_text = best_match_faq['answer_hi']
            elif 'answer_en' in best_match_faq:
                bot_response_text = best_match_faq['answer_en']
            else:
                bot_response_text = "I found a relevant answer, but it's not available in your selected language. Here's the English version: " + best_match_faq.get('answer_en', 'No answer available.')
            status_text = "answered"
        else:
            bot_response_text = "I'm sorry, I don't have a precise answer for that right now. Your query has been noted for review by our team."
            status_text = "unanswered"
            # TODO: Trigger email escalation here (Week 3/4 task)
            # from backend.services.email_service import send_escalation_email
            # await send_escalation_email(user_query_text, user_id, language, similarity_score)

    except Exception as e:
        logger.error(f"Error processing chat query '{user_query_text}': {e}", exc_info=True)
        bot_response_text = "An internal error occurred while processing your request. Please try again."
        status_text = "error"
        raise HTTPException(status_code=500, detail="Internal server error.")
    finally:
        # 5. Log the interaction
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