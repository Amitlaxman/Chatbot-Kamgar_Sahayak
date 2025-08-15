# apk_router.py

from fastapi import APIRouter
from fastapi.responses import FileResponse
import os

router = APIRouter()

APK_DIRECTORY = "apks"

@router.get("/download")
async def download_apk():
    apk_name = "app-release.apk"
    file_path = os.path.join(APK_DIRECTORY, apk_name)
    
    # Check if the file exists
    if os.path.exists(file_path):
        return FileResponse(
            path=file_path,
            media_type="application/vnd.android.package-archive",
            filename=apk_name
        )
    else:
        return {"error": "APK not found"}