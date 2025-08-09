from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from typing import List, Dict, Any
import logging
import os
from backend.db.mongo_utils import get_mongo_db, get_unanswered_logs, get_all_logs_entries, get_admin_user, create_admin_user
from backend.models.chat_model import AdminLogin
from passlib.context import CryptContext # For password hashing
from datetime import datetime, timedelta
import jwt # PyJWT for token handling

logger = logging.getLogger(__name__)
router = APIRouter()

# --- Security Setup (Basic for College Project) ---
# In a real production app, use proper JWT libraries like `python-jose`
# and more robust token management. This is a simplified example.

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
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
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
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        user = await get_admin_user(username)
        if user is None:
            raise credentials_exception
        return user
    except jwt.PyJWTError:
        raise credentials_exception


# --- Admin Dashboard Endpoints ---
@router.get("/unanswered_logs", response_model=List[Dict[str, Any]])
async def get_unanswered_logs_api(current_user: dict = Depends(get_current_admin_user)):
    """Get all unanswered queries/logs."""
    try:
        logs = await get_unanswered_logs()
        return logs
    except Exception as e:
        logger.error(f"Error fetching unanswered logs: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to fetch unanswered logs")

@router.get("/all_logs", response_model=List[Dict[str, Any]])
async def get_all_logs_api(current_user: dict = Depends(get_current_admin_user)):
    """Get all query logs."""
    try:
        logs = await get_all_logs_entries()
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
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["username"], "role": user["role"]}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@router.post("/register_admin")
async def register_admin_user(user_data: AdminLogin):
    """
    Endpoint to register a new admin user.
    FOR DEVELOPMENT ONLY - REMOVE/PROTECT IN PRODUCTION!
    """
    existing_user = await get_admin_user(user_data.username)
    if existing_user:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Username already registered")
    
    hashed_password = get_password_hash(user_data.password)
    new_user_data = {"username": user_data.username, "hashed_password": hashed_password, "role": "admin"}
    try:
        await create_admin_user(new_user_data)
        return {"message": "Admin user registered successfully"}
    except Exception as e:
        logger.error(f"Error registering admin user: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to register admin user.")

# --- Admin Data Endpoints (Protected) ---

@router.get("/unanswered_queries", response_model=List[Dict[str, Any]])
async def get_admin_unanswered_queries(current_user: Dict[str, Any] = Depends(get_current_admin_user)):
    """Retrieves all unanswered queries for admin review (requires authentication)."""
    if current_user["role"] not in ["admin", "viewer"]: # Example role check
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized to view this resource.")
    
    try:
        queries = await get_unanswered_logs()
        return queries
    except Exception as e:
        logger.error(f"Failed to retrieve unanswered queries for admin: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve data.")

@router.get("/logs", response_model=List[Dict[str, Any]])
async def get_admin_all_logs(current_user: Dict[str, Any] = Depends(get_current_admin_user)):
    """Retrieves all interaction logs (requires authentication)."""
    if current_user["role"] not in ["admin", "viewer"]: # Example role check
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized to view this resource.")
    
    try:
        logs = await get_all_logs_entries()
        return logs
    except Exception as e:
        logger.error(f"Failed to retrieve all logs for admin: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve data.")

# Example of a protected endpoint that only 'admin' role can access
@router.post("/add_faq")
async def add_faq_entry(faq_data: Dict[str, Any], current_user: Dict[str, Any] = Depends(get_current_admin_user)):
    """
    Adds a new FAQ entry (requires 'admin' role).
    This would typically trigger an ETL process or direct DB insert.
    For simplicity, this is a placeholder.
    """
    if current_user["role"] != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Only admins can add FAQs.")
    
    # TODO: Implement actual FAQ insertion logic here, potentially calling ingest_faqs.py or mongo_utils
    logger.info(f"Admin {current_user['username']} attempting to add FAQ: {faq_data.get('question_id')}")
    return {"message": "FAQ addition endpoint (placeholder) reached."}
