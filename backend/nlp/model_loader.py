import logging
from sentence_transformers import SentenceTransformer
from typing import Optional

logger = logging.getLogger(__name__)

# Global variable to hold the loaded model
_embedding_model: Optional[SentenceTransformer] = None

def load_nlp_model(model_name: str):
    """
    Loads the NLP model into memory and stores it in a global variable.
    """
    global _embedding_model
    if _embedding_model is not None:
        logger.info("NLP model is already loaded.")
        return

    try:
        logger.info(f"Loading NLP model: {model_name}")
        _embedding_model = SentenceTransformer(model_name)
        logger.info("NLP model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load NLP model: {e}", exc_info=True)
        _embedding_model = None
        raise

def get_embedding_model() -> SentenceTransformer:
    """
    Returns the pre-loaded NLP model.
    Raises an error if the model has not been loaded.
    """
    global _embedding_model
    if _embedding_model is None:
        logger.error("Attempted to get embedding model before it was loaded.")
        raise RuntimeError("NLP model is not loaded. Please ensure startup has completed.")
    return _embedding_model