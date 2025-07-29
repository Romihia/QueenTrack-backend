from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from app.routes.video_routes import router as video_router
from app.routes.events_routes import router as events_router
from app.routes.system_routes import router as system_router
from app.routes.test_routes import router as test_router
from fastapi.middleware.cors import CORSMiddleware
import os
import logging
import asyncio
from typing import List
from pathlib import Path

# Create necessary directories first
os.makedirs('/data/logs', exist_ok=True)
os.makedirs('/data/videos', exist_ok=True)

# Configure logging after creating directories with error handling
try:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('/data/logs/queen_track.log'),
            logging.StreamHandler()
        ]
    )
except Exception as e:
    # Fallback to console-only logging if file logging fails
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    print(f"Warning: Could not set up file logging: {e}")

logger = logging.getLogger(__name__)

# Log startup message
logger.info("🚀 Queen Track Backend v2.0 initializing...")

# Initialize session-based services
try:
    from app.services import (
        camera_session_manager,
        websocket_connection_manager,
        dual_camera_websocket_handler,
        event_coordinator
    )
    logger.info("✅ Session-based services initialized successfully")
    logger.info("🎯 Camera Session Manager ready")
    logger.info("🔗 WebSocket Connection Manager ready")
    logger.info("📹 Dual Camera WebSocket Handler ready")
    logger.info("⚡ Event Coordinator ready")
except Exception as e:
    logger.error(f"💥 Error initializing session services: {e}")
    raise

app = FastAPI(
    title="Queen Track Backend",
    version="2.0.0",
    description="Professional API for Queen Track bee monitoring system with dual camera session management and event coordination."
)

# Startup event handler for settings synchronization
@app.on_event("startup")
async def startup_event():
    """Initialize and sync settings on startup"""
    try:
        logger.info("🔄 Starting settings synchronization on startup...")
        
        # Import here to avoid circular imports
        try:
            from app.services.settings_service import get_settings_service
            from app.services.email_service import get_email_service
            
            # Initialize and sync settings
            settings_service = await get_settings_service()
            await settings_service.sync_settings_on_startup()
            
            # Load email settings from database
            email_service = get_email_service()
            await email_service.load_settings_from_database()
            
            logger.info("✅ Settings synchronized successfully on startup")
            
        except ImportError as e:
            logger.warning(f"⚠️ Settings service not available during startup: {e}")
        except Exception as e:
            logger.error(f"❌ Error during startup settings sync: {e}")
        
    except Exception as e:
        logger.error(f"❌ Critical error during startup: {e}")
        # Don't raise - let the server start even if settings sync fails

# Videos directory already created above
videos_dir = "/data/videos"

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
    allow_origin_regex=r".*",
)

# הוספת הראוטים
app.include_router(video_router, prefix="/video", tags=["video"])
app.include_router(events_router, prefix="/events", tags=["events"])
app.include_router(system_router, prefix="/system", tags=["system"])
app.include_router(test_router, prefix="/test", tags=["testing"])

# Log successful initialization
logger.info("✅ All routers loaded successfully")
logger.info("📧 Email service ready")
logger.info("🎥 Video service ready")
logger.info("🔍 System monitoring ready")
logger.info("🐝 Queen Track Backend v2.0 ready to serve!")

@app.get("/")
def root():
    logger.info("Root endpoint accessed")
    return {
        "message": "Queen Track Backend v2.0 is running",
        "features": [
            "Email notifications",
            "Video streaming with range support", 
            "Real camera integration",
            "Professional logging",
            "System health monitoring"
        ],
        "endpoints": {
            "system_health": "/system/health",
            "test_email": "/video/test-email",
            "video_streaming": "/video/videos/{file_path}",
            "bee_monitoring": "/video/live-stream"
        }
    }

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "Queen Track Backend v2.0", "timestamp": "2024-12-16"}

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
