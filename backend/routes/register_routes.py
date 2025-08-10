from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

class RegisterData(BaseModel):
    name: str
    address: str
    workType: str

@router.post("/register")
def register_labour(data: RegisterData):
    # For now, just print the data (later, you can save it to DB)
    print("Registered Labour:", data.dict())
    return {"message": "Registration successful!"}
