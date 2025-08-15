from fastapi import APIRouter, HTTPException, Depends, status, Body
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from bson import ObjectId 
from datetime import datetime, timedelta
from typing import List, Dict, Any
import logging
import os
from backend.db.mongo_utils import get_mongo_db, get_unanswered_logs, get_all_logs_entries, get_admin_user, create_admin_user
from backend.models.chat_model import AdminLogin
from passlib.context import CryptContext  # For password hashing
import jwt  # PyJWT for token handling

logger = logging.getLogger(__name__)
router = APIRouter()

# --- Security Setup ---
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-super-secret-key-please-change-this-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/admin_api/token")


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

    try:
        db = await get_mongo_db()
        result = await db["queries"].update_one(
            {"_id": ObjectId(query_id)},
            {"$set": {"answer": answer_text, "status": "answered", "answeredAt": datetime.utcnow()}},
        )
        if result.modified_count == 0:
            raise HTTPException(status_code=404, detail="Query not found or already answered")
        return {"message": "Answer submitted successfully"}
    except Exception as e:
        logger.error(f"Error submitting answer: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to submit answer")
