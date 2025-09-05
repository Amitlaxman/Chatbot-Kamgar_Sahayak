from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from backend.models.chat_model import ChatQuery, ChatResponse, LogEntry
from backend.db.mongo_utils import get_legal_db, get_chatbot_db, insert_log_entry
from backend.nlp.rag import retrieve_relevant_faqs, format_context_for_generation
from backend.nlp.model_loader import get_embedding_model
import logging
import os
import io
import pyttsx3
import base64
from datetime import datetime
from typing import Dict, Any, List, Optional
import re
import traceback

# Import cosine_similarity from similarity module
from backend.nlp.similarity import cosine_similarity

logger = logging.getLogger(__name__)
router = APIRouter()

# Configuration
CONFIDENCE_THRESHOLD = 0.1  # lowered threshold to increase recall
KEYWORDS_COLLECTION = os.getenv("KEYWORDS_COLLECTION", "keywords")

# --- Updated ChatResponse model to include optional audio data ---
class ChatResponse(BaseModel):
    bot_response: str = Field(..., description="The chatbot's response text.")
    status: str = Field(..., description="Status of the query (e.g., 'answered', 'unanswered', 'error').")
    language: str = Field(..., description="Language of the bot's response.")
    query_id: Optional[str] = Field(None, description="Optional ID for the processed query.")
    similarity_score: Optional[float] = Field(None, description="Cosine similarity score if answered by NLP.")
    audio_data: Optional[str] = Field(None, description="Base64 encoded audio of the bot's response.")

async def get_synonyms_from_db(query_text: str, language: str) -> List[str]:
    '''Fetch synonyms from the chatbot database's keywords collection.'''
    try:
        db = get_chatbot_db()
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

        expanded_keywords = [kw for kw in set(expanded_keywords) if kw]
        return expanded_keywords
    except Exception as e:
        logger.error(f"Error fetching synonyms: {e}")
        return []

class ChatRequest(BaseModel):
    user_id: str = Field(...)
    query_text: str = Field(...)
    language: str = Field("en")
    chat_history: Optional[List[Dict[str, str]]] = None

def generate_tts_audio(text: str, lang: str) -> Optional[str]:
    '''Generates a Base64 encoded audio string from text using pyttsx3 (offline).'''
    try:
        engine = pyttsx3.init()
        # Set language/voice if needed (pyttsx3 uses system voices)
        # For Hindi, you may need to set a Hindi voice if available
        voices = engine.getProperty('voices')
        if lang == 'hi':
            for voice in voices:
                if 'hi' in voice.languages or 'Hindi' in voice.name:
                    engine.setProperty('voice', voice.id)
                    break
        else:
            for voice in voices:
                if 'en' in voice.languages or 'English' in voice.name:
                    engine.setProperty('voice', voice.id)
                    break
        audio_stream = io.BytesIO()
        engine.save_to_file(text, 'temp_tts_output.mp3')
        engine.runAndWait()
        with open('temp_tts_output.mp3', 'rb') as f:
            audio_base64 = base64.b64encode(f.read()).decode('utf-8')
        os.remove('temp_tts_output.mp3')
        return audio_base64
    except Exception as e:
        logger.error(f"Failed to generate TTS audio: {e}", exc_info=True)
        return None

@router.post("/chat", response_model=ChatResponse)
async def chat_with_bot(request: ChatRequest):
    user_query_text = request.query_text
    user_id = request.user_id
    language = request.language
    chat_history = request.chat_history

    logger.info(f"Received chat query from {user_id} ({language}): '{user_query_text}'")

    context_window = 5
    history_context = ""
    if chat_history:
        for msg in chat_history[-context_window:]:
            prefix = "User:" if msg.get("sender") == "user" else "Bot:"
            history_context += f"{prefix} {msg.get('text', '')}\n"

    bot_response_text = ""
    status_text = "unanswered"
    similarity_score = None
    audio_data = None

    try:
        # 1. Expand query with synonyms
        synonym_keywords = await get_synonyms_from_db(user_query_text, language)
        logger.info(f"Synonyms for query '{user_query_text}': {synonym_keywords}")
        expanded_query_text = user_query_text + " " + " ".join(synonym_keywords)

        # 2. RAG: Retrieve top relevant chunks
        rag_query = (history_context + "User: " + expanded_query_text).strip()
        top_chunks = retrieve_relevant_faqs(rag_query, top_k=3)
        context_str = format_context_for_generation(top_chunks, language=language)

        if top_chunks:
            embedding_model = get_embedding_model() # Get the pre-loaded model
            if not embedding_model:
                raise Exception("Embedding model is not loaded.")
                
            query_emb = embedding_model.encode(rag_query)
            scored_chunks = []
            use_hi = bool(re.search(r'[\u0900-\u097F]', rag_query))
            emb_field = 'embedding_hi' if use_hi else 'embedding_en'
            
            for idx, chunk in enumerate(top_chunks):
                chunk_emb = chunk.get(emb_field)
                if not chunk_emb:
                    text = (chunk.get('content_hi') if use_hi else chunk.get('content_en')) or chunk.get('source', '')
                    if text:
                        try:
                            chunk_emb = embedding_model.encode(text)
                        except Exception as e:
                            logger.warning(f"Failed to generate fallback embedding for chunk idx={idx}: {e}")
                            continue
                
                if chunk_emb is not None:
                    sim = cosine_similarity(query_emb, chunk_emb)
                    scored_chunks.append((sim, chunk))
            
            if scored_chunks:
                scored_chunks.sort(reverse=True, key=lambda x: x[0])
                best_score, best_chunk = scored_chunks[0]
                similarity_score = float(best_score)
                logger.info(f"Selected chunk with score {similarity_score}")

                if best_chunk:
                    if language == 'hi':
                        bot_response_text = best_chunk.get('content_hi') or best_chunk.get('content_en') or best_chunk.get('source', 'उत्तर उपलब्ध नहीं है।')
                    else:
                        bot_response_text = best_chunk.get('content_en') or best_chunk.get('content_hi') or best_chunk.get('source', 'Answer not available.')
                
                status_text = "answered" if similarity_score >= CONFIDENCE_THRESHOLD else "low_confidence"
            else:
                bot_response_text = ('I am sorry, I could not find a relevant chunk. Your query has been logged for review.' if language != 'hi' else 'माफ़ कीजिए, मैं उपयुक्त जानकारी नहीं ढूँढ पाया। आपका प्रश्न समीक्षा के लिए लॉग कर दिया गया है।')
                status_text = "unanswered"

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

        # Generate TTS audio for the bot's text response
        audio_data = generate_tts_audio(bot_response_text, lang=language)

    except Exception as e:
        logger.error(f"Error processing chat query '{user_query_text}': {e}\n{traceback.format_exc()}")
        bot_response_text = "An internal error occurred while processing your request. Please try again."
        status_text = "error"
        audio_data = None
        
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
        similarity_score=similarity_score,
        audio_data=audio_data
    )