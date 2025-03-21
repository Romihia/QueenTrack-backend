from fastapi import APIRouter, WebSocket, UploadFile, File, HTTPException
import cv2
import numpy as np
import time
import os
import boto3
from starlette.websockets import WebSocketDisconnect
from app.core.config import settings
from ultralytics import YOLO

router = APIRouter()

# Initialize YOLO models
model_detect = YOLO("yolov8n.pt")  # Detection model
model_classify = YOLO("best.pt")   # Classification model

def upload_to_s3(file_path):
    """
    Upload file to S3
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

def format_time(milliseconds):
    """
    Convert milliseconds to HH:MM:SS.msec format
    """
    total_seconds = milliseconds / 1000.0
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = (total_seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"

def process_frame(frame):
    """
    Process frame for bee detection and classification
    """
    # Get current timestamp
    time_str = format_time(time.time() * 1000)  # Using system time as an example

    # Step 1: Detect bees using YOLO detection model
    results_detect = model_detect(frame)
    
    if results_detect and len(results_detect) > 0 and hasattr(results_detect[0], 'boxes'):
        boxes = results_detect[0].boxes.xyxy
        
        for box in boxes:
            x1, y1, x2, y2 = map(int, box[:4])
            cropped = frame[y1:y2, x1:x2]

            # Step 2: Classify the detected bee
            results_class = model_classify.predict(source=cropped, verbose=False)
            
            if len(results_class) > 0:
                r = results_class[0]
                if r.probs is not None:
                    cls_id = int(r.probs.top1)
                    cls_conf = float(r.probs.top1conf)
                    cls_name = r.names[cls_id]
                else:
                    cls_name = "Unknown"
                    cls_conf = 0.0
            else:
                cls_name = "Unknown"
                cls_conf = 0.0

            # Draw rectangle and label
            label = f"{cls_name} {cls_conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Add timestamp to frame
    cv2.putText(frame, time_str, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    
    return frame

@router.websocket("/live-stream")
async def live_stream(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket connected")

    timestamp = int(time.time())
    os.makedirs("./temp_videos", exist_ok=True)
    video_filename = f"./temp_videos/stream_{timestamp}.webm"
    processed_filename = f"./temp_videos/processed_stream_{timestamp}.mp4"
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    try:
        # שלב 1: שמירת הזרם לקובץ
        with open(video_filename, "wb") as f:
            while True:
                data = await websocket.receive_bytes()
                print(f"Received {len(data)} bytes")
                f.write(data)
                
    except WebSocketDisconnect:
        print("WebSocket disconnected")
    except Exception as e:
        print(f"Error in WebSocket: {e}")
    finally:
        # שלב 2: עיבוד הקובץ השלם
        if os.path.exists(video_filename) and os.path.getsize(video_filename) > 0:
            print(f"Processing video: {video_filename}")
            cap = cv2.VideoCapture(video_filename)
            if not cap.isOpened():
                print("Failed to open video file - ensure OpenCV is built with FFmpeg support")
            else:
                fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                if width > 0 and height > 0:
                    out = cv2.VideoWriter(processed_filename, fourcc, fps, (width, height))
                    frame_count = 0
                    
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        processed_frame = process_frame(frame)
                        out.write(processed_frame)
                        frame_count += 1
                    
                    cap.release()
                    out.release()
                    print(f"Processed {frame_count} frames")
                    
                    if frame_count > 0:
                        print(f"Uploading {processed_filename} to S3...")
                        upload_to_s3(processed_filename)
                    else:
                        print("No frames processed")
                else:
                    print("Invalid video dimensions")
        
        # ניקוי
        if os.path.exists(video_filename):
            os.remove(video_filename)
            print(f"Cleaned up temporary file: {video_filename}")

@router.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No file provided")

    try:
        os.makedirs("./uploaded_videos", exist_ok=True)
        os.makedirs("./processed_videos", exist_ok=True)

        input_path = f"./uploaded_videos/{file.filename}"
        output_path = f"./processed_videos/processed_{file.filename}"

        # Save uploaded file
        with open(input_path, "wb") as f:
            f.write(await file.read())

        # Process video
        cap = cv2.VideoCapture(input_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            processed_frame = process_frame(frame)
            out.write(processed_frame)

        cap.release()
        out.release()

        # Upload processed video to S3
        upload_to_s3(output_path)

        return {"status": "success", "message": "Video processed and uploaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
		
	