from backend.db.mongo_utils import get_legal_db

# Utility to fetch reference link for a collection
def get_collection_reference_link(collection_name):
    """
    Returns the reference link for a given collection from the 'links' collection in the main legal_db.
    """
    db = get_legal_db()
    col = db['links']
    entry = col.find_one({'collection': collection_name})
    if entry:
        return entry.get('reference_link', '')
    return ''
