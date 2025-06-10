from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from app.routes.video_routes import router as video_router
from app.routes.events_routes import router as events_router
from fastapi.middleware.cors import CORSMiddleware
import os
from typing import List
from pathlib import Path

app = FastAPI(
    title="Bee Vision Backend",
    version="0.1.0",
    description="API for processing video streams and events."
)

# Create videos directory if it doesn't exist
videos_dir = "/data/videos"
os.makedirs(videos_dir, exist_ok=True)

# Create subdirectories for video organization
video_subdirs = [
    "outside_videos",
    "temp_videos", 
    "uploaded_videos",
    "processed_videos"
]

for subdir in video_subdirs:
    os.makedirs(f"{videos_dir}/{subdir}", exist_ok=True)

# Mount static files for video serving
app.mount("/videos", StaticFiles(directory=videos_dir), name="videos")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# הוספת הראוטים
app.include_router(video_router, prefix="/video", tags=["video"])
app.include_router(events_router, prefix="/events", tags=["events"])

@app.get("/")
def root():
    return {"message": "Bee Vision Backend is running"}

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "QueenTrack Backend"}

@app.get("/videos/list")
def list_videos(folder: str = None):
    """
    List all available videos, optionally filtered by folder.
    Access: http://[SERVER_IP]:8000/videos/list
    """
    try:
        base_path = Path(videos_dir)
        videos = []
        
        if folder:
            # List videos in specific folder
            folder_path = base_path / folder
            if not folder_path.exists():
                raise HTTPException(status_code=404, detail=f"Folder '{folder}' not found")
            search_path = folder_path
        else:
            # List all videos in all folders
            search_path = base_path
        
        # Find all video files
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
        
        for ext in video_extensions:
            for video_file in search_path.rglob(f"*{ext}"):
                relative_path = video_file.relative_to(base_path)
                file_size = video_file.stat().st_size
                modified_time = video_file.stat().st_mtime
                
                # Convert path separators for URL compatibility
                url_path = str(relative_path).replace("\\", "/")
                
                videos.append({
                    "filename": video_file.name,
                    "path": url_path,
                    "url": f"/videos/{url_path}",
                    "size_bytes": file_size,
                    "size_mb": round(file_size / (1024 * 1024), 2),
                    "modified_timestamp": modified_time,
                    "folder": str(relative_path.parent) if relative_path.parent != Path('.') else "root"
                })
        
        # Sort by modification time (newest first)
        videos.sort(key=lambda x: x['modified_timestamp'], reverse=True)
        
        return {
            "total_videos": len(videos),
            "folder": folder or "all",
            "videos": videos,
            "available_folders": video_subdirs,
            "access_info": {
                "base_url": "/videos/",
                "example_access": f"http://[SERVER_IP]:8000/videos/outside_videos/video_1234567890.mp4",
                "list_by_folder": "Add ?folder=outside_videos to filter by folder"
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing videos: {str(e)}")

@app.get("/videos/folders")
def list_video_folders():
    """
    List all available video folders.
    Access: http://[SERVER_IP]:8000/videos/folders
    """
    try:
        base_path = Path(videos_dir)
        folders = []
        
        for item in base_path.iterdir():
            if item.is_dir():
                video_count = len(list(item.glob("*.mp4")) + list(item.glob("*.avi")) + 
                                list(item.glob("*.mov")) + list(item.glob("*.mkv")))
                
                folders.append({
                    "name": item.name,
                    "path": f"/videos/{item.name}/",
                    "video_count": video_count,
                    "list_url": f"/videos/list?folder={item.name}"
                })
        
        return {
            "folders": folders,
            "total_folders": len(folders),
            "access_info": {
                "folder_access": "http://[SERVER_IP]:8000/videos/[FOLDER_NAME]/",
                "list_videos": "http://[SERVER_IP]:8000/videos/list"
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing folders: {str(e)}")
