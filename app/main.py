from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from app.routes.video_routes import router as video_router
from app.routes.events_routes import router as events_router
from fastapi.middleware.cors import CORSMiddleware
import os

app = FastAPI(
    title="Bee Vision Backend",
    version="0.1.0",
    description="API for processing video streams and events."
)

# Create videos directory if it doesn't exist
videos_dir = "/data/videos"
os.makedirs(videos_dir, exist_ok=True)

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
