from fastapi import APIRouter, WebSocket, UploadFile, File, HTTPException, Body
import cv2
import numpy as np
import time
import os
import json
from starlette.websockets import WebSocketDisconnect
from ultralytics import YOLO
from datetime import datetime
from app.services.service import create_event, update_event, get_all_events
import threading
from pydantic import BaseModel

# Camera configuration model
class CameraConfig(BaseModel):
    internal_camera_id: str
    external_camera_id: str

router = APIRouter()

# Initialize YOLO models
model_detect = YOLO("yolov8n.pt")  # Detection model
model_classify = YOLO("best.pt")   # Classification model

# Define the Region of Interest (ROI) for the hive entrance
# Format: [x_min, y_min, x_max, y_max] as a rectangle
HIVE_ENTRANCE_ROI = [200, 300, 400, 450]  # Example values - adjust based on your camera setup

# Video storage configuration
VIDEOS_DIR = "/data/videos"
os.makedirs(VIDEOS_DIR, exist_ok=True)

# Create subdirectories for different video types
os.makedirs(f"{VIDEOS_DIR}/outside_videos", exist_ok=True)
os.makedirs(f"{VIDEOS_DIR}/temp_videos", exist_ok=True)
os.makedirs(f"{VIDEOS_DIR}/uploaded_videos", exist_ok=True)
os.makedirs(f"{VIDEOS_DIR}/processed_videos", exist_ok=True)

# Bee tracking state (simple version for a single bee)
bee_state = {
    "previous_status": None,  # "inside" or "outside"
    "current_event_id": None,  # To track the current ongoing event in the database
    "current_status": None,    # Current bee status for external queries
}

# External camera state
external_camera = {
    "is_recording": False,
    "video_writer": None,
    "capture": None,
    "thread": None,
    "should_stop": False,
    "output_file": None,
    "stream_url": None,        # URL for accessing the camera stream (if available)
}

# Camera configuration
camera_config = {
    "internal_camera_id": None,
    "external_camera_id": None,
}

@router.post("/camera-config")
async def set_camera_config(config: CameraConfig):
    """
    Save camera configuration from the frontend
    """
    camera_config["internal_camera_id"] = config.internal_camera_id
    camera_config["external_camera_id"] = config.external_camera_id
    
    print(f"Camera configuration updated: Internal camera ID: {config.internal_camera_id}, External camera ID: {config.external_camera_id}")
    
    return {"status": "success", "message": "Camera configuration saved successfully"}

def save_video_locally(file_path):
    """
    Save video locally and return the URL for accessing it
    """
    try:
        # Get relative path from the videos directory
        videos_dir = "/data/videos"
        
        # If the file is already in the videos directory, create relative path
        if file_path.startswith(videos_dir):
            relative_path = file_path[len(videos_dir):].lstrip('/')
            local_url = f"/videos/{relative_path}"
        else:
            # For files outside the videos directory, just use the filename
            filename = os.path.basename(file_path)
            local_url = f"/videos/{filename}"
        
        print(f"Video saved locally: {file_path}")
        print(f"Video accessible at: {local_url}")
        
        return local_url
    except Exception as e:
        print(f"Failed to process local video: {e}")
        return None

def format_time(milliseconds):
    """
    Convert milliseconds to HH:MM:SS.msec format
    """
    total_seconds = milliseconds / 1000.0
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = (total_seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"

def is_inside_roi(x_center, y_center, roi=HIVE_ENTRANCE_ROI):
    """
    Check if the given point is inside the ROI
    """
    x_min, y_min, x_max, y_max = roi
    return (x_min <= x_center <= x_max) and (y_min <= y_center <= y_max)

# ---- External Camera Control Functions ----

def start_external_camera():
    """
    Start recording with the external camera.
    This is a mock implementation that can be replaced with actual camera control code.
    
    Returns:
        str: Path to the output video file
    """
    if external_camera["is_recording"]:
        print("External camera is already recording")
        return None
    
    # Create output directory if it doesn't exist
    outside_videos_dir = f"{VIDEOS_DIR}/outside_videos"
    os.makedirs(outside_videos_dir, exist_ok=True)
    
    # Generate output filename with timestamp
    timestamp = int(time.time())
    output_path = f"{outside_videos_dir}/video_outside_{timestamp}.mp4"
    external_camera["output_file"] = output_path
    
    # Check if we have a configured external camera
    if camera_config["external_camera_id"]:
        print(f"[INFO] Starting external camera (ID: {camera_config['external_camera_id']}) recording to {output_path}")
        
        try:
            # In a real implementation with connected physical camera:
            # This would use the device ID selected in the frontend
            # cap = cv2.VideoCapture(1)  # Example for a second camera
            # 
            # For web cameras, it could use device ID or URL:
            # cap = cv2.VideoCapture(camera_config["external_camera_id"])
            
            # For now, just log that we're using the configured camera
            print(f"[MOCK] Using configured external camera: {camera_config['external_camera_id']}")
            
            # Set mock stream - in a real implementation, this might be a URL to a streaming server
            external_camera["stream_url"] = f"/videos/outside_videos/stream_{timestamp}.m3u8"  # Example
        except Exception as e:
            print(f"Error starting external camera: {e}")
            # Fall back to mock implementation if there's an error
    else:
        print(f"[MOCK] No external camera configured, using mock camera")
    
    # For simulation purposes, we'll create a blank video with timestamps
    # In a real implementation, we would capture from the configured camera
    mock_width, mock_height = 640, 480
    try:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, 20.0, (mock_width, mock_height))
    except Exception as e:
        print(f"[ERROR] Failed to initialize video writer: {e}")
        return f"Error: Failed to initialize video writer - {str(e)}"
    
    external_camera["video_writer"] = video_writer
    external_camera["is_recording"] = True
    external_camera["should_stop"] = False
    
    # Start recording thread
    thread = threading.Thread(target=mock_recording_thread)
    thread.daemon = True
    thread.start()
    
    external_camera["thread"] = thread
    
    return output_path

def stop_external_camera():
    """
    Stop recording with the external camera and return the path to the recorded video.
    
    Returns:
        str: Path to the recorded video file or None if no recording was happening
    """
    if not external_camera["is_recording"]:
        print("External camera is not recording")
        return None
    
    print("[MOCK] Stopping external camera recording")
    
    # Signal the recording thread to stop
    external_camera["should_stop"] = True
    
    # Wait for the thread to finish
    if external_camera["thread"] is not None:
        external_camera["thread"].join(timeout=5.0)
    
    # Clean up resources
    if external_camera["video_writer"] is not None:
        external_camera["video_writer"].release()
    
    output_file = external_camera["output_file"]
    
    # Reset camera state
    external_camera["is_recording"] = False
    external_camera["video_writer"] = None
    external_camera["thread"] = None
    external_camera["stream_url"] = None
    external_camera["output_file"] = None
    
    print(f"[MOCK] External camera recording stopped, saved to {output_file}")
    
    return output_file

def mock_recording_thread():
    """
    Thread function that simulates recording from an external camera
    by creating frames with timestamps.
    """
    mock_width, mock_height = 640, 480
    start_time = time.time()
    
    # Get camera info for display
    camera_id = camera_config["external_camera_id"] or "No camera configured"
    
    while not external_camera["should_stop"]:
        # Create a blank frame (black background)
        frame = np.zeros((mock_height, mock_width, 3), dtype=np.uint8)
        
        # Add timestamp
        elapsed = time.time() - start_time
        timestamp_text = f"Outside Camera - {elapsed:.2f}s"
        cv2.putText(frame, timestamp_text, (20, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        # Add camera info
        camera_text = f"Camera ID: {camera_id}"
        cv2.putText(frame, camera_text, (20, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Add bee status
        bee_status = bee_state["current_status"] or "Unknown"
        status_text = f"Bee Status: {bee_status}"
        cv2.putText(frame, status_text, (20, 130),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Add "RECORDING" indicator
        cv2.putText(frame, "RECORDING", (20, mock_height - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Write the frame
        if external_camera["video_writer"] is not None:
            external_camera["video_writer"].write(frame)
        
        # Sleep to simulate real-time recording (approximately 20 FPS)
        time.sleep(0.05)

# ---- End External Camera Functions ----

def process_frame(frame):
    """
    Process frame for bee detection and classification

    Returns:
        tuple: (processed_frame, bee_status, current_time) where:
            - processed_frame is the frame with visualizations
            - bee_status is None or "inside"/"outside"
            - current_time is the timestamp for the frame
    """
    # Get current timestamp
    time_str = format_time(time.time() * 1000)
    current_time = datetime.now()
    bee_detected = False
    bee_status = None

    # Create a copy of the frame to process and avoid modifying the original
    processed_frame = frame.copy()

    try:
        # Step 1: Detect bees using YOLO detection model
        results_detect = model_detect(processed_frame)
    except Exception as e:
        print(f"[ERROR] YOLO detection failed: {e}")
        # Return original frame with error overlay
        cv2.putText(processed_frame, f"Detection Error: {str(e)[:50]}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return processed_frame, None, current_time
    
    if results_detect and len(results_detect) > 0 and hasattr(results_detect[0], 'boxes'):
        boxes = results_detect[0].boxes.xyxy
        
        for box in boxes:
            x1, y1, x2, y2 = map(int, box[:4])
            cropped = processed_frame[y1:y2, x1:x2]

            # Step 2: Classify the detected bee
            results_class = model_classify.predict(source=cropped, verbose=False)
            
            if len(results_class) > 0:
                r = results_class[0]
                if r.probs is not None:
                    cls_id = int(r.probs.top1)
                    cls_conf = float(r.probs.top1conf)
                    cls_name = r.names[cls_id]
                    
                    # If it's our marked bee (update this condition based on your classification model)
                    if cls_name == "marked_bee" and cls_conf > 0.7:  # Adjust threshold as needed
                        bee_detected = True
                        
                        # Calculate center point of the bounding box
                        x_center = (x1 + x2) // 2
                        y_center = (y1 + y2) // 2
                        
                        # Check if the bee is inside or outside the ROI
                        if is_inside_roi(x_center, y_center):
                            bee_status = "inside"
                        else:
                            bee_status = "outside"
                        
                        # Store current status for tracking and external queries
                        bee_state["current_status"] = bee_status
                        
                        # Draw ROI on frame
                        cv2.rectangle(processed_frame, 
                                    (HIVE_ENTRANCE_ROI[0], HIVE_ENTRANCE_ROI[1]), 
                                    (HIVE_ENTRANCE_ROI[2], HIVE_ENTRANCE_ROI[3]), 
                                    (0, 255, 255), 2)
                        
                        # Draw rectangle around the bee and show status
                        color = (0, 255, 0) if bee_status == "inside" else (0, 165, 255)
                        cv2.rectangle(processed_frame, (x1, y1), (x2, y2), color, 2)
                        label = f"{cls_name} ({bee_status})"
                        cv2.putText(processed_frame, label, (x1, y1 - 5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                else:
                    cls_name = "Unknown"
                    cls_conf = 0.0
            else:
                cls_name = "Unknown"
                cls_conf = 0.0

            # Draw rectangle and label for other detected objects
            if not bee_detected:
                label = f"{cls_name} {cls_conf:.2f}"
                cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(processed_frame, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # If no marked bee is detected in this frame but we had a previous state
    if not bee_detected and bee_state["previous_status"] is not None:
        # Draw the ROI regardless
        cv2.rectangle(processed_frame, 
                    (HIVE_ENTRANCE_ROI[0], HIVE_ENTRANCE_ROI[1]), 
                    (HIVE_ENTRANCE_ROI[2], HIVE_ENTRANCE_ROI[3]), 
                    (0, 255, 255), 2)
    
    # Add timestamp to frame
    cv2.putText(processed_frame, time_str, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    
    # Add external camera status indicator if it's recording
    if external_camera["is_recording"]:
        cv2.putText(processed_frame, "External Camera: RECORDING", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Instead of setting attributes on the frame object, we'll just return them
    # Make bee_status None if no bee was detected
    if not bee_detected:
        bee_status = None
        
    # Add these as attributes for backward compatibility
    # This is safer than directly setting on numpy array
    class FrameInfo:
        pass
    
    frame_info = FrameInfo()
    frame_info.bee_status = bee_status
    frame_info.current_time = current_time
    
    return processed_frame, bee_status, current_time

async def handle_bee_tracking(current_status, current_time):
    """
    Handle the tracking logic and database operations
    """
    previous_status = bee_state["previous_status"]
    
    # First frame or after reset
    if previous_status is None:
        bee_state["previous_status"] = current_status
        return
    
    # Detect transitions
    if previous_status == "inside" and current_status == "outside":
        # Bee has exited the hive
        print(f"[{current_time}] Event detected: BEE EXIT")
        
        # Start recording with the external camera
        output_path = start_external_camera()
        
        # Create a new exit event
        event_data = {
            "time_out": current_time,
            "time_in": None,  # Will be updated when bee returns
            "video_url": None  # Will be updated with video URL later
        }
        new_event = await create_event(event_data)
        bee_state["current_event_id"] = new_event.id
        
    elif previous_status == "outside" and current_status == "inside":
        # Bee has entered the hive
        print(f"[{current_time}] Event detected: BEE ENTRANCE")
        
        # Stop the external camera if it was recording
        video_path = stop_external_camera()
        video_url = None
        
        # If a video was recorded, save locally and get URL
        if video_path and os.path.exists(video_path):
            video_url = save_video_locally(video_path)
        
        # If we have an ongoing event, update it
        if bee_state["current_event_id"]:
            event_update = {
                "time_in": current_time,
                "video_url": video_url
            }
            await update_event(bee_state["current_event_id"], event_update)
            bee_state["current_event_id"] = None
    
    # Update the state
    bee_state["previous_status"] = current_status

@router.get("/external-camera-status")
async def get_external_camera_status():
    """
    Return the current status of the external camera and bee information
    """
    return {
        "is_recording": external_camera["is_recording"],
        "stream_url": external_camera["stream_url"],
        "last_bee_status": bee_state["current_status"],
        "internal_camera_id": camera_config["internal_camera_id"],
        "external_camera_id": camera_config["external_camera_id"]
    }

@router.websocket("/live-stream")
async def live_stream(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket connected")

    timestamp = int(time.time())
    temp_videos_dir = f"{VIDEOS_DIR}/temp_videos"
    os.makedirs(temp_videos_dir, exist_ok=True)
    processed_filename = f"{temp_videos_dir}/processed_stream_{timestamp}.mp4"

    video_writer = None
    frame_count = 0

    try:
        while True:
            data = await websocket.receive_bytes()
            np_arr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if frame is None:
                print("Invalid frame received")
                continue

            # Process the frame and store bee_status as a separate variable instead of as attribute
            processed_frame, bee_status, current_time = process_frame(frame)
            
            # Check if bee was detected and handle tracking
            if bee_status is not None:
                try:
                    await handle_bee_tracking(bee_status, current_time)
                    
                    # Send status update to the client
                    status_update = {
                        "bee_status": bee_status,
                        "external_camera_status": external_camera["is_recording"]
                    }
                    await websocket.send_text(json.dumps(status_update))
                except Exception as e:
                    print(f"Error in bee tracking: {e}")

            if video_writer is None:
                height, width = processed_frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                fps = 20.0 
                video_writer = cv2.VideoWriter(processed_filename, fourcc, fps, (width, height))
                if not video_writer.isOpened():
                    raise Exception("Failed to open VideoWriter")

            video_writer.write(processed_frame)
            frame_count += 1
            print(f"Processed frame {frame_count}")

    except WebSocketDisconnect:
        print("WebSocket disconnected")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if video_writer:
            video_writer.release()
            print(f"Saved processed video: {processed_filename}")
            
            # Save video locally and get URL
            video_url = save_video_locally(processed_filename)
            
            # If we have an ongoing event and the video was successfully saved, update the event
            if bee_state["current_event_id"] and video_url:
                event_update = {"video_url": video_url}
                await update_event(bee_state["current_event_id"], event_update)
            
            # Make sure to stop external camera if it's still recording    
            if external_camera["is_recording"]:
                stop_external_camera()
                
            # Reset bee tracking state
            bee_state["previous_status"] = None
            bee_state["current_event_id"] = None

@router.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No file provided")

    try:
        uploaded_videos_dir = f"{VIDEOS_DIR}/uploaded_videos"
        processed_videos_dir = f"{VIDEOS_DIR}/processed_videos"
        os.makedirs(uploaded_videos_dir, exist_ok=True)
        os.makedirs(processed_videos_dir, exist_ok=True)

        input_path = f"{uploaded_videos_dir}/{file.filename}"
        output_path = f"{processed_videos_dir}/processed_{file.filename}"

        # Save uploaded file
        with open(input_path, "wb") as f:
            f.write(await file.read())

        # Reset tracking state for uploaded video processing
        bee_state["previous_status"] = None
        bee_state["current_event_id"] = None

        # Process video
        cap = cv2.VideoCapture(input_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            processed_frame, bee_status, current_time = process_frame(frame)
            
            # Handle tracking for uploaded videos - use getattr to safely get attributes
            if bee_status is not None:
                try:
                    await handle_bee_tracking(bee_status, current_time)
                except Exception as e:
                    print(f"Error in bee tracking during upload: {e}")
                
            out.write(processed_frame)
            frame_count += 1
            if frame_count % 30 == 0:  # Log every 30 frames
                print(f"Processed frame {frame_count}")

        cap.release()
        out.release()

        # Make sure external camera is stopped at the end of processing
        if external_camera["is_recording"]:
            stop_external_camera()

        # Save processed video locally and get URL
        video_url = save_video_locally(output_path)
        
        # If we have an ongoing event and the video was saved, update the event
        if bee_state["current_event_id"] and video_url:
            event_update = {"video_url": video_url}
            await update_event(bee_state["current_event_id"], event_update)
            
        # Reset tracking state
        bee_state["previous_status"] = None
        bee_state["current_event_id"] = None

        return {"status": "success", "message": "Video processed and saved locally", "video_url": video_url}
    except Exception as e:
        # Make sure to stop external camera in case of exceptions
        if external_camera["is_recording"]:
            stop_external_camera()
        raise HTTPException(status_code=500, detail=str(e))
		
	