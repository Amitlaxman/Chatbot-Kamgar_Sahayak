

from typing import List, Dict, Any
import os
import json
from backend.nlp.similarity import get_embedding, cosine_similarity


def retrieve_relevant_faqs(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """
    Retrieve the top_k most relevant chunks from Minimum_Wages_Act_1948.json using embedding similarity.
    """
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'Minimum_Wages_Act_1948.json'))
    with open(data_path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    if not chunks:
        return []
    query_emb = get_embedding(query)
    scored = []
    for chunk in chunks:
        # Use both English and Hindi content for retrieval
        text = (chunk.get('content_en', '') or '') + ' ' + (chunk.get('content_hi', '') or '')
        emb = get_embedding(text)
        sim = cosine_similarity(query_emb, emb)
        scored.append((sim, chunk))
    scored.sort(reverse=True, key=lambda x: x[0])
    return [chunk for sim, chunk in scored[:top_k]]


def format_context_for_generation(faqs: List[Dict[str, Any]], language: str = 'en') -> str:
    """
    Format retrieved chunks as context for a generative model.
    """
    lines = []
    for i, chunk in enumerate(faqs, 1):
        if language == 'hi':
            text = chunk.get('content_hi', '')
        else:
            text = chunk.get('content_en', '')
        lines.append(f"Chunk {i}:\n{text}\n")
    return '\n'.join(lines)

# You can add more RAG utilities here as needed.
