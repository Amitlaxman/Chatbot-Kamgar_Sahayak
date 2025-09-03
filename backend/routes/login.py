
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, EmailStr
from backend.db.mongo_utils import get_user_by_email, create_user, verify_user
from passlib.context import CryptContext
import jwt
import os
from datetime import datetime, timedelta

router = APIRouter()

SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-super-secret-key-please-change-this-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class LoginData(BaseModel):
    email: EmailStr
    password: str

class RegisterData(BaseModel):
    name: str
    email: EmailStr
    password: str
    address: str = ""
    workType: str = ""

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

@router.post("/login")
async def login_user(data: LoginData):
    valid = await verify_user(data.email, data.password)
    if not valid:
        raise HTTPException(status_code=401, detail="Invalid email or password")
    user = await get_user_by_email(data.email)
    access_token = create_access_token({"sub": user["email"], "role": "user"})
    return {"message": "Login successful", "token": access_token, "user": {"name": user["name"], "email": user["email"]}}

@router.post("/register-user")
async def register_user(data: RegisterData):
    existing_user = await get_user_by_email(data.email)
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    user_data = {
        "name": data.name,
        "email": data.email,
        "password": data.password,
        "address": data.address,
        "workType": data.workType
    }
    await create_user(user_data)
    return {"message": "User registered successfully"}
