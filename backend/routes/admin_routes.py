from fastapi import APIRouter, HTTPException, Depends, status, Body
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from bson import ObjectId 
from datetime import datetime, timedelta
from typing import List, Dict, Any
import logging
import os
import jwt
from backend.models.chat_model import AdminLogin
from passlib.context import CryptContext

# Import the correct database utility functions
from backend.db.mongo_utils import (
    get_chatbot_db,
    get_admin_db,
    get_unanswered_logs,
    get_all_logs_entries,
    get_admin_user,
    create_admin_user,
    insert_admin_marking,
    insert_admin_answer
)

logger = logging.getLogger(__name__)
router = APIRouter()

from backend.services.email_service import send_email

# --- Security Setup ---
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-super-secret-key-please-change-this-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/admin_api/token")

async def get_current_admin_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
        user = await get_admin_user(email)
        if user is None:
            raise credentials_exception
        return user
    except jwt.PyJWTError:
        raise credentials_exception

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta if expires_delta else timedelta(minutes=15))
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# --- New: Mail Unanswered Queries Endpoint ---
@router.post("/mail_unanswered")
async def mail_unanswered_queries(current_user: dict = Depends(get_current_admin_user)):
    try:
        unanswered = get_unanswered_logs()
        if not unanswered:
            return {"message": "No unanswered queries found."}
        # Format email body
        body = "Unanswered Queries:\n\n"
        for q in unanswered:
            body += f"ID: {q.get('_id', q.get('id', ''))}\nQuestion: {q.get('question', '')}\nAsked By: {q.get('user', '')}\nDate: {q.get('date', '')}\n---\n"
        subject = "Unanswered Queries Report"
        to_email = "kaaamgar.sahayak@gmail.com"
        success = send_email(subject, body, to_email)
        logger.info(f"send_email returned: {success}")
        if success:
            return {"message": f"Unanswered queries mailed to {to_email}"}
        else:
            raise HTTPException(status_code=500, detail="Failed to send email. Check logs for details.")
    except Exception as e:
        logger.error(f"Failed to mail unanswered queries: {e}")
        raise HTTPException(status_code=500, detail="An internal error occurred while mailing queries.")

# --- Utility: Add admin user with known password if not present ---
@router.post("/force_add_admin")
async def force_add_admin():
    email = "kaaamgar.sahayak@gmail.com"
    password = "Passswoord"
    user = await get_admin_user(email)
    if user:
        return {"message": "Admin already exists."}
    
    hashed_password = get_password_hash(password)
    admin_data = {"email": email, "hashed_password": hashed_password, "role": "admin"}
    await create_admin_user(admin_data)
    return {"message": f"Admin {email} created with password '{password}'"}

@router.get("/unanswered_logs", response_model=List[Dict[str, Any]])
async def get_unanswered_logs_api(current_user: dict = Depends(get_current_admin_user)):
    try:
        logs = get_unanswered_logs()
        return logs
    except Exception as e:
        logger.error(f"Error fetching unanswered logs: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to fetch unanswered logs")

@router.get("/all_logs", response_model=List[Dict[str, Any]])
async def get_all_logs_api(current_user: dict = Depends(get_current_admin_user)):
    try:
        logs = get_all_logs_entries()
        return logs
    except Exception as e:
        logger.error(f"Error fetching all logs: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to fetch all logs")

@router.post("/token")
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = await get_admin_user(form_data.username)
    if not user or not verify_password(form_data.password, user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["email"], "role": user["role"]}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@router.post("/register_admin")
async def register_admin_user(user_data: AdminLogin):
    existing_user = await get_admin_user(user_data.email)
    if existing_user:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email already registered")

    hashed_password = get_password_hash(user_data.password)
    new_user_data = {"email": user_data.email, "hashed_password": hashed_password, "role": "admin"}
    try:
        await create_admin_user(new_user_data)
        return {"message": "Admin user registered successfully"}
    except Exception as e:
        logger.error(f"Error registering admin user: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to register admin user.")

@router.get("/unanswered_queries", response_model=List[Dict[str, Any]])
async def get_admin_unanswered_queries(current_user: Dict[str, Any] = Depends(get_current_admin_user)):
    if current_user["role"] not in ["admin", "viewer"]:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized to view this resource.")
    try:
        queries = get_unanswered_logs()
        return queries
    except Exception as e:
        logger.error(f"Failed to retrieve unanswered queries for admin: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve data.")

@router.get("/logs", response_model=List[Dict[str, Any]])
async def get_admin_all_logs(current_user: Dict[str, Any] = Depends(get_current_admin_user)):
    if current_user["role"] not in ["admin", "viewer"]:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized to view this resource.")
    try:
        logs = get_all_logs_entries()
        return logs
    except Exception as e:
        logger.error(f"Failed to retrieve all logs for admin: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve data.")

@router.post("/add_faq")
async def add_faq_entry(faq_data: Dict[str, Any], current_user: Dict[str, Any] = Depends(get_current_admin_user)):
    if current_user["role"] != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Only admins can add FAQs.")
    logger.info(f"Admin {current_user['email']} attempting to add FAQ: {faq_data.get('question_id')}")
    # TODO: Add actual FAQ insertion logic here
    return {"message": "FAQ addition endpoint (placeholder) reached."}

@router.post("/answer/{query_id}")
async def submit_answer(
    query_id: str,
    answer_data: Dict[str, Any] = Body(...),
    current_user: Dict[str, Any] = Depends(get_current_admin_user),
):
    if current_user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Only admins can submit answers")
    
    answer_text = answer_data.get("answer")
    if not answer_text or answer_text.strip() == "":
        raise HTTPException(status_code=400, detail="Answer cannot be empty")
    
    # Correctly call the database utility functions
    try:
        success = await insert_admin_answer(query_id, answer_text)
        if not success:
            raise HTTPException(status_code=404, detail="Query not found or already answered")
        return {"message": "Answer submitted successfully"}
    except Exception as e:
        logger.error(f"Error submitting answer: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to submit answer")

@router.post("/mark_query/{query_id}")
async def mark_query(
    query_id: str,
    mark_data: Dict[str, Any] = Body(...),
    current_user: Dict[str, Any] = Depends(get_current_admin_user),
):
    """
    Mark a query as Irrelevant, Answerable, or Pending.
    """
    if current_user["role"] != "admin":
        logger.error(f"Mark query failed: user {current_user['email']} is not admin.")
        raise HTTPException(status_code=403, detail="Only admins can mark queries")
    
    mark_status = mark_data.get("status")
    if mark_status not in ["Irrelevant", "Answerable", "Pending"]:
        logger.error(f"Mark query failed: invalid status {mark_status}")
        raise HTTPException(status_code=400, detail="Invalid mark status")
    
    try:
        # Correctly call the database utility function
        success = await insert_admin_marking(query_id, {"status": mark_status, "marked_by": current_user["email"], "marked_at": datetime.utcnow()})
        if not success:
            return {"message": "Query already marked or not found."}
            
        logger.info(f"Marking inserted for query {query_id}")
        return {"message": "Query marked successfully"}
    except Exception as e:
        logger.error(f"Error marking query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to mark query")

@router.post("/answer_query/{query_id}")
async def answer_query(
    query_id: str,
    answer_data: Dict[str, Any] = Body(...),
    current_user: Dict[str, Any] = Depends(get_current_admin_user),
):
    """
    Submit an answer for a query. Stores in admin_answers collection with extracted keywords.
    """
    if current_user["role"] != "admin":
        logger.error(f"Answer query failed: user {current_user['email']} is not admin.")
        raise HTTPException(status_code=403, detail="Only admins can answer queries")
        
    answer_text = answer_data.get("answer")
    if not answer_text or answer_text.strip() == "":
        logger.error(f"Answer query failed: empty answer for query {query_id}")
        raise HTTPException(status_code=400, detail="Answer cannot be empty")
        
    # Correctly call the database utility function
    try:
        success = await insert_admin_answer(query_id, answer_text)
        if not success:
             return {"message": "Answer already submitted or query not found."}
        
        logger.info(f"Answer inserted for query {query_id}")
        return {"message": "Answer submitted and logged successfully"}
    except Exception as e:
        logger.error(f"Error submitting admin answer: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to submit admin answer")

@router.get("/marked_queries", response_model=List[Dict[str, Any]])
async def get_marked_queries(current_user: Dict[str, Any] = Depends(get_current_admin_user)):
    admin_db = get_admin_db()
    marked_raw = list(admin_db["admin_marking"].find({}, {"_id": 0, "query_id": 1, "query_log": 1}))
    marked = []
    for m in marked_raw:
        marked.append({
            "query_id": m.get("query_id"),
            "question": (m.get("query_log", {}).get("question") or m.get("query_log", {}).get("query_text") or "")
        })
    return marked