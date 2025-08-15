from fastapi import FastAPI
from fastapi.responses import FileResponse
import uvicorn
import os

app = FastAPI()

# Specify the directory where your APK files are stored
APK_DIRECTORY = "apks"

@app.get("/download")
async def download_apk():
    """
    Endpoint to serve the specific APK file named 'app-release.apk'.
    """
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

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)