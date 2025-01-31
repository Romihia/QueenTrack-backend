from fastapi import APIRouter, WebSocket, UploadFile, File, HTTPException
import cv2
import numpy as np
import time
import os
import boto3
from starlette.websockets import WebSocketDisconnect
from app.core.config import settings


router = APIRouter()
def upload_to_s3(file_path):
    """
    upLoad to S3
    """
    s3 = boto3.client(
        "s3",
        aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
        region_name=settings.AWS_S3_REGION
    )

    bucket_name = settings.AWS_S3_BUCKET_NAME
    s3_key = f"data_bee/{os.path.basename(file_path)}"

    try:
        s3.upload_file(file_path, bucket_name, s3_key)
        print(f"Upload Successful: {s3_key}")
        os.remove(file_path) 
    except Exception as e:
        print(f"Failed to upload video: {e}")

@router.websocket("/live-stream")
async def live_stream(websocket: WebSocket):
 
    print("Attempting to accept WebSocket connection")
    await websocket.accept()

    timestamp = int(time.time())
    video_filename = f"./temp_videos/stream_{timestamp}.mp4"
    os.makedirs("./temp_videos", exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    frame_width, frame_height = 640, 480
    fps = 20.0
    video_writer = cv2.VideoWriter(video_filename, fourcc, fps, (frame_width, frame_height))

    try:
        while True:
            data = await websocket.receive_bytes()
            nparr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is not None:
                video_writer.write(frame)

    except WebSocketDisconnect:
        print("WebSocket disconnected. Saving and uploading video...")
        video_writer.release()
        upload_to_s3(video_filename)

    except Exception as e:
        video_writer.release()
        print(f"Error during streaming: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    

@router.post("/upload")
async def upload_video(file: UploadFile = File(...)):
  
    if not file:
        raise HTTPException(status_code=400, detail="No file provided")

    try:
        os.makedirs("./uploaded_videos", exist_ok=True)

        file_path = f"./uploaded_videos/{file.filename}"

        with open(file_path, "wb") as f:
            f.write(await file.read())

        upload_to_s3(file_path)

        return {"status": "success", "message": "Video uploaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



def process_frame(frame):
    """
    פונקציה לעיבוד פריימים (לדוגמה: זיהוי דבורים מסומנות)
    """
    print("streaming")

