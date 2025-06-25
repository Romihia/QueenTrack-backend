"""
Video Routes - מתקן לגמרי עם שירותים חדשים
"""
from fastapi import APIRouter, WebSocket, UploadFile, File, HTTPException, Body, Request
import cv2
import numpy as np
import time
import os
import json
from starlette.websockets import WebSocketDisconnect
from ultralytics import YOLO
from datetime import datetime
from app.services.service import create_event, update_event, get_all_events
from app.schemas.schema import EventCreate, EventUpdate
from app.services.email_service import email_service
from app.services.video_service import video_service
from app.services.video_format_converter import video_format_converter
from app.services.video_recording_service import video_recording_service
from app.services.bee_tracking_service import bee_tracking_service
import threading
from pydantic import BaseModel
import logging
import asyncio
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CameraConfig(BaseModel):
    internal_camera_id: str
    external_camera_id: str

class ProcessingSettings(BaseModel):
    # Detection and Classification Settings
    detection_enabled: bool = True
    classification_enabled: bool = True
    detection_confidence_threshold: float = 0.25
    classification_confidence_threshold: float = 0.3
    computer_vision_fallback: bool = True
    
    # Drawing Settings
    draw_bee_path: bool = True
    draw_center_line: bool = True
    draw_roi_box: bool = True
    draw_status_text: bool = True
    draw_confidence_scores: bool = True
    draw_timestamp: bool = True
    draw_frame_counter: bool = False
    
    # Path and Tracking Settings
    path_history_size: int = 50
    min_consecutive_detections: int = 3
    transition_cooldown: float = 2.0
    
    # Computer Vision Settings
    cv_contour_min_area: int = 50
    cv_contour_max_area: int = 2000
    cv_aspect_ratio_min: float = 0.3
    cv_aspect_ratio_max: float = 3.0
    
    # Video Management Settings
    auto_delete_videos_after_session: bool = False
    keep_original_videos: bool = True
    auto_convert_videos: bool = True
    video_buffer_seconds: int = 5
    
    # ROI Settings
    roi_x_min: int = 640
    roi_y_min: int = 0
    roi_x_max: int = 1280
    roi_y_max: int = 720
    
    # Email Notification Settings
    email_notifications_enabled: bool = True
    email_on_exit: bool = True
    email_on_entrance: bool = True

class SystemSettings(BaseModel):
    processing: ProcessingSettings = ProcessingSettings()
    camera_config: Dict[str, str] = {"internal_camera_id": "0", "external_camera_id": "1"}

# Global settings instance
current_settings = SystemSettings()

router = APIRouter()

# Initialize YOLO model - unified detection & classification
model_detect = YOLO("yolov8n.pt")  # Your trained model that detects regular_bee (0) and marked_bee (1)
# Removed model_classify - no longer needed since detection model handles classification

# Video storage configuration
VIDEOS_DIR = "/data/videos"
os.makedirs(VIDEOS_DIR, exist_ok=True)

# Camera configuration
camera_config = {
    "internal_camera_id": None,
    "external_camera_id": None,
}

# Set up video conversion callback
def video_conversion_callback(event_id: str, internal_path: str, external_path: str):
    """Callback להמרת וידאו לאחר סיום הקלטה - sync version"""
    try:
        logger.info(f"🎬 Starting video conversion for event {event_id}")
        
        # Update conversion status to processing
        import asyncio
        asyncio.run(update_conversion_status(event_id, "processing"))
        
        # Convert videos using the format converter
        conversion_results = video_format_converter.convert_event_videos(
            internal_path, external_path
        )
        
        if conversion_results["conversion_success"]:
            logger.info(f"✅ Video conversion completed for event {event_id}")
            
            # Get URLs for converted videos
            internal_converted_url = None
            external_converted_url = None
            
            if conversion_results["internal_converted"]:
                internal_converted_url = video_format_converter.get_video_url_from_path(
                    conversion_results["internal_converted"]
                )
            
            if conversion_results["external_converted"]:
                external_converted_url = video_format_converter.get_video_url_from_path(
                    conversion_results["external_converted"]
                )
            
            logger.info(f"📹 Converted videos for event {event_id}:")
            logger.info(f"   Internal: {internal_converted_url}")
            logger.info(f"   External: {external_converted_url}")
            
            # Update the database with converted video URLs
            asyncio.run(update_converted_videos(
                event_id, 
                internal_converted_url, 
                external_converted_url
            ))
            
        else:
            # Conversion failed
            error_message = "; ".join(conversion_results["errors"]) if conversion_results["errors"] else "Unknown conversion error"
            logger.error(f"❌ Video conversion failed for event {event_id}: {error_message}")
            
            # Update conversion status to failed
            asyncio.run(update_conversion_status(event_id, "failed", error_message))
            
    except Exception as e:
        logger.error(f"💥 Video conversion callback error for event {event_id}: {e}")
        # Update conversion status to failed
        try:
            import asyncio
            asyncio.run(update_conversion_status(event_id, "failed", str(e)))
        except Exception:
            pass

# Helper functions for database updates
async def update_conversion_status(event_id: str, status: str, error_message: str = None):
    """Update conversion status in database"""
    try:
        update_data = EventUpdate(
            conversion_status=status,
            conversion_error=error_message if error_message else None
        )
        await update_event(event_id, update_data)
        logger.info(f"📝 Updated conversion status for event {event_id}: {status}")
    except Exception as e:
        logger.error(f"Failed to update conversion status for event {event_id}: {e}")

async def update_converted_videos(event_id: str, internal_url: str = None, external_url: str = None):
    """Update converted video URLs in database"""
    try:
        update_data = EventUpdate(
            internal_video_url_converted=internal_url,
            external_video_url_converted=external_url,
            conversion_status="completed"
        )
        await update_event(event_id, update_data)
        logger.info(f"📝 Updated converted video URLs for event {event_id}")
        logger.info(f"   Internal converted: {internal_url}")
        logger.info(f"   External converted: {external_url}")
    except Exception as e:
        logger.error(f"Failed to update converted video URLs for event {event_id}: {e}")

# Set the conversion callback
video_recording_service.set_conversion_callback(video_conversion_callback)

# Bee tracking state - פשוט ובסיסי
bee_state = {
    "current_status": None,
    "status_sequence": [],
    "consecutive_detections": {"inside": 0, "outside": 0},
    "event_active": False,
    "current_event_id": None,
    "position_history": []
}

# Configuration constants
FRAME_WIDTH = 1280  # Default frame width
FRAME_HEIGHT = 720  # Default frame height
CENTER_LINE_X = FRAME_WIDTH // 2  # Vertical center line
MIN_CONSECUTIVE_DETECTIONS = 3
POSITION_HISTORY_SIZE = 1000  # Keep 1000 points instead of 50

def is_inside_hive(x_center, y_center):
    """Check if bee is inside the hive ROI using configurable settings"""
    roi = [
        current_settings.processing.roi_x_min,
        current_settings.processing.roi_y_min,
        current_settings.processing.roi_x_max,
        current_settings.processing.roi_y_max
    ]
    return roi[0] <= x_center <= roi[2] and roi[1] <= y_center <= roi[3]

def detect_sequence_pattern(x_center, y_center, current_time):
    """
    Detect crossing events for triggering
    Returns: (bee_status, event_action)
    """
    # Determine current raw status
    is_currently_inside = is_inside_hive(x_center, y_center)
    current_status = "inside" if is_currently_inside else "outside"
    
    # Update consecutive detection counters
    if current_status == "inside":
        bee_state["consecutive_detections"]["inside"] += 1
        bee_state["consecutive_detections"]["outside"] = 0
    else:
        bee_state["consecutive_detections"]["outside"] += 1
        bee_state["consecutive_detections"]["inside"] = 0
    
    # Check if we have enough consecutive detections
    consecutive_inside = bee_state["consecutive_detections"]["inside"]
    consecutive_outside = bee_state["consecutive_detections"]["outside"]
    
    logger.info(f"Consecutive counts: inside={consecutive_inside}, outside={consecutive_outside}")
    
    # Only update status if we have enough consecutive detections
    confirmed_status = None
    if consecutive_inside >= current_settings.processing.min_consecutive_detections:
        confirmed_status = "inside"
    elif consecutive_outside >= current_settings.processing.min_consecutive_detections:
        confirmed_status = "outside"
    
    if confirmed_status:
        # Update sequence only when we have a confirmed status change
        if bee_state["current_status"] != confirmed_status:
            previous_status = bee_state["current_status"]
            bee_state["current_status"] = confirmed_status
            bee_state["status_sequence"].append(confirmed_status)
            
            # Keep only last 10 statuses
            if len(bee_state["status_sequence"]) > 10:
                bee_state["status_sequence"] = bee_state["status_sequence"][-10:]
            
            logger.info(f"Status change: {previous_status} → {confirmed_status}")
            logger.info(f"Full sequence: {' → '.join(bee_state['status_sequence'])}")
            logger.info(f"Event active: {bee_state['event_active']}")
            
            # Check for crossing events
            if previous_status and previous_status != confirmed_status:
                if not bee_state["event_active"]:
                    # Check for event start: inside → outside (bee exits hive)
                    if previous_status == "inside" and confirmed_status == "outside":
                        bee_state["event_active"] = True
                        logger.warning("🚪 EVENT START: Bee crossed from inside → outside")
                        return confirmed_status, "start_event"
                else:
                    # Check for event end: outside → inside (bee returns to hive)
                    if previous_status == "outside" and confirmed_status == "inside":
                        bee_state["event_active"] = False
                        logger.warning("🏠 EVENT END: Bee crossed from outside → inside")
                        return confirmed_status, "end_event"
    
    return bee_state.get("current_status"), None

def format_time(milliseconds):
    """Convert milliseconds to HH:MM:SS.msec format"""
    total_seconds = milliseconds / 1000.0
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = (total_seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"

def draw_center_line_info(frame):
    """Draw center line and ROI information if enabled"""
    if not current_settings.processing.draw_center_line:
        return frame
        
    height, width = frame.shape[:2]
    center_x = width // 2
    
    # Draw vertical center line
    cv2.line(frame, (center_x, 0), (center_x, height), (0, 255, 255), 2)
    
    if current_settings.processing.draw_roi_box:
        # Draw ROI rectangle
        roi_color = (255, 0, 255)  # Magenta
        cv2.rectangle(frame, 
                     (current_settings.processing.roi_x_min, current_settings.processing.roi_y_min),
                     (current_settings.processing.roi_x_max, current_settings.processing.roi_y_max),
                     roi_color, 2)
        
        # Add ROI label
        cv2.putText(frame, "HIVE ENTRANCE ROI", 
                   (current_settings.processing.roi_x_min, current_settings.processing.roi_y_min - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, roi_color, 2)
    
    if current_settings.processing.draw_status_text:
        # Add center line label
        cv2.putText(frame, "CENTER", (center_x + 10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Add inside/outside labels
        cv2.putText(frame, "INSIDE HIVE", (current_settings.processing.roi_x_min + 10, 
                   current_settings.processing.roi_y_min + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.putText(frame, "OUTSIDE HIVE", (50, height - 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    return frame

def draw_bee_path(frame):
    """Draw bee tracking path if enabled"""
    if not current_settings.processing.draw_bee_path:
        return frame
        
    debug_info = bee_tracking_service.get_debug_info()
    position_history = debug_info.get("position_history", [])
    
    if len(position_history) < 2:
        return frame
    
    # Draw path lines
    for i in range(1, len(position_history)):
        prev_x, prev_y, prev_time, prev_status = position_history[i-1]
        curr_x, curr_y, curr_time, curr_status = position_history[i]
        
        # Color based on status
        color = (0, 255, 0) if curr_status == "inside" else (0, 165, 255)
        
        cv2.line(frame, (int(prev_x), int(prev_y)), (int(curr_x), int(curr_y)), color, 2)
        
        # Draw position points
        cv2.circle(frame, (int(curr_x), int(curr_y)), 3, color, -1)
    
    # Draw current position with larger circle
    if position_history:
        last_x, last_y, last_time, last_status = position_history[-1]
        color = (0, 255, 0) if last_status == "inside" else (0, 165, 255)
        cv2.circle(frame, (int(last_x), int(last_y)), 8, color, 3)
        
        if current_settings.processing.draw_status_text:
            cv2.putText(frame, f"CURRENT: {last_status.upper()}", 
                       (int(last_x) + 15, int(last_y) - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    return frame

def find_bee_location_in_frame(frame):
    """Find bee location using computer vision with configurable parameters"""
    if not current_settings.processing.computer_vision_fallback:
        return None, None
        
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Adaptive thresholding
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY_INV, 11, 2)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area and aspect ratio using configurable settings
        potential_bees = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if current_settings.processing.cv_contour_min_area <= area <= current_settings.processing.cv_contour_max_area:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h
                if current_settings.processing.cv_aspect_ratio_min <= aspect_ratio <= current_settings.processing.cv_aspect_ratio_max:
                    potential_bees.append((contour, area, x + w//2, y + h//2))
        
        if potential_bees:
            # Return the largest contour (most likely to be a bee)
            largest_bee = max(potential_bees, key=lambda x: x[1])
            return largest_bee[2], largest_bee[3]  # x, y center
        
        return None, None
        
    except Exception as e:
        logger.error(f"Computer vision bee detection failed: {e}")
        return None, None

def process_frame(frame):
    """Process video frame with configurable settings"""
    current_time = datetime.now()
    bee_status = None
    
    try:
        # Add frame counter if enabled
        if current_settings.processing.draw_frame_counter:
            global frame_counter
            if 'frame_counter' not in globals():
                frame_counter = 0
            frame_counter += 1
            cv2.putText(frame, f"Frame: {frame_counter}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add timestamp if enabled
        if current_settings.processing.draw_timestamp:
            timestamp_text = current_time.strftime("%H:%M:%S.%f")[:-3]
            cv2.putText(frame, timestamp_text, (10, frame.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # YOLO Detection (if enabled)
        marked_bee_detected = False
        bee_x, bee_y = None, None
        detection_confidence = 0.0
        
        if current_settings.processing.detection_enabled:
            try:
                results = model_detect(frame, conf=current_settings.processing.detection_confidence_threshold)
                
                if results and len(results) > 0:
                    result = results[0]
                    
                    if hasattr(result, 'boxes') and result.boxes is not None:
                        boxes = result.boxes
                        
                        if hasattr(boxes, 'xyxy') and len(boxes.xyxy) > 0:
                            for i in range(len(boxes.xyxy)):
                                try:
                                    cls_id = int(boxes.cls[i])
                                    conf = float(boxes.conf[i])
                                    
                                    # Check if it's a marked bee with sufficient confidence
                                    if (cls_id == 1 and 
                                        conf >= current_settings.processing.classification_confidence_threshold):
                                        
                                        x1, y1, x2, y2 = boxes.xyxy[i][:4]
                                        bee_x = int((x1 + x2) / 2)
                                        bee_y = int((y1 + y2) / 2)
                                        detection_confidence = conf
                                        marked_bee_detected = True
                                        
                                        # Draw bounding box
                                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                                        
                                        if current_settings.processing.draw_confidence_scores:
                                            label = f"marked_bee {conf:.2f}"
                                            cv2.putText(frame, label, (int(x1), int(y1) - 10),
                                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                                        
                                        break
                                        
                                except Exception as e:
                                    logger.error(f"Error processing detection box {i}: {e}")
                                    continue
                                    
            except Exception as e:
                logger.error(f"YOLO detection error: {e}")
        
        # Computer Vision Fallback (if enabled and no YOLO detection)
        if (current_settings.processing.computer_vision_fallback and 
            not marked_bee_detected):
            
            cv_x, cv_y = find_bee_location_in_frame(frame)
            if cv_x is not None and cv_y is not None:
                bee_x, bee_y = cv_x, cv_y
                marked_bee_detected = True
                detection_confidence = 0.5  # Default confidence for CV detection
                
                # Draw CV detection indicator
                cv2.circle(frame, (bee_x, bee_y), 15, (255, 0, 255), 3)
                if current_settings.processing.draw_confidence_scores:
                    cv2.putText(frame, "CV Detection", (bee_x + 20, bee_y - 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        
        # Determine bee status if detected
        if marked_bee_detected and bee_x is not None and bee_y is not None:
            if is_inside_hive(bee_x, bee_y):
                bee_status = "inside"
            else:
                bee_status = "outside"
            
            # Add bee to tracking service
            bee_tracking_service.add_position_to_history(bee_x, bee_y, bee_status, current_time)
        
        # Draw visual elements based on settings
        if current_settings.processing.draw_center_line:
            frame = draw_center_line_info(frame)
        if current_settings.processing.draw_bee_path:
            frame = draw_bee_path(frame)
        
        # Add debug information if enabled
        if current_settings.processing.draw_status_text:
            debug_info = bee_tracking_service.get_debug_info()
            current_status = debug_info["bee_state"]["current_status"]
            event_active = debug_info["bee_state"]["event_active"]
            
            status_text = f"Status: {current_status or 'Unknown'}"
            event_text = f"Event: {'ACTIVE' if event_active else 'Inactive'}"
            
            cv2.putText(frame, status_text, (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, event_text, (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        return frame, bee_status, current_time, bee_x, bee_y
        
    except Exception as e:
        logger.error(f"Error in process_frame: {e}")
        return frame, None, current_time, None, None

async def handle_bee_event(event_action, current_status, current_time, bee_image=None):
    """Handle bee events: start event, end event"""
    logger.info(f"🎯 handle_bee_event called: action={event_action}, status={current_status}")
    
    if event_action == "start_event":
        logger.warning(f"🚪 [{current_time}] EVENT STARTED: Bee exited after entering")
        
        try:
            event_data = EventCreate(
                time_out=current_time,
                time_in=None,
                internal_video_url=None,
                external_video_url=None,
                conversion_status="pending"  # Initial status
            )
            new_event = await create_event(event_data)
            bee_state["current_event_id"] = new_event.id
            logger.info(f"📝 Created new event with ID: {new_event.id}")
            
            # Start video recording
            video_paths = video_recording_service.start_event_recording(new_event.id)
            if video_paths:
                # Update event with initial video paths
                update_data = EventUpdate(
                    internal_video_url=video_paths.get("internal_video"),
                    external_video_url=video_paths.get("external_video")
                )
                await update_event(new_event.id, update_data)
                logger.info(f"📹 Video recording started for event {new_event.id}")
            
        except Exception as e:
            logger.error(f"💥 Error creating event: {str(e)}")
        
        # Send email notification if enabled
        if current_settings.processing.email_notifications_enabled and current_settings.processing.email_on_exit:
            try:
                additional_info = {
                    "center_line_x": CENTER_LINE_X,
                    "event_type": "event_start",
                    "crossing_direction": "inside_to_outside",
                    "detection_time": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "sequence_description": "Bee crossed from inside hive to outside area"
                }
                
                email_success = email_service.send_bee_detection_notification(
                    event_type="exit",
                    timestamp=current_time,
                    bee_image=bee_image,
                    additional_info=additional_info
                )
                
                if email_success:
                    logger.info("✅ Event start notification email sent successfully")
                else:
                    logger.error("❌ Failed to send event start notification email")
                    
            except Exception as e:
                logger.error(f"💥 Error sending event start email notification: {str(e)}")
            
    elif event_action == "end_event":
        logger.warning(f"🏠 [{current_time}] EVENT ENDED: Bee returned to hive")
        
        try:
            current_event_id = bee_state["current_event_id"]
            if current_event_id:
                # Stop video recording
                video_paths = video_recording_service.stop_event_recording()
                
                # Update event with end time and final video paths
                event_update = EventUpdate(
                    time_in=current_time,
                    internal_video_url=video_paths.get("internal_video"),
                    external_video_url=video_paths.get("external_video")
                )
                await update_event(current_event_id, event_update)
                logger.info(f"📝 Updated event {current_event_id} with end time and video paths")
                logger.info(f"🎬 Video conversion will start automatically")
                
                bee_state["current_event_id"] = None
                
        except Exception as e:
            logger.error(f"💥 Error updating event: {str(e)}")
        
        # Send email notification for bee entrance if enabled
        if current_settings.processing.email_notifications_enabled and current_settings.processing.email_on_entrance:
            try:
                additional_info = {
                    "center_line_x": CENTER_LINE_X,
                    "event_type": "event_end",
                    "crossing_direction": "outside_to_inside",
                    "detection_time": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "sequence_description": "Bee returned from outside to inside hive"
                }
                
                email_success = email_service.send_bee_detection_notification(
                    event_type="entrance",
                    timestamp=current_time,
                    bee_image=bee_image,
                    additional_info=additional_info
                )
                
                if email_success:
                    logger.info("✅ Event end notification email sent successfully")
                else:
                    logger.error("❌ Failed to send event end notification email")
                    
            except Exception as e:
                logger.error(f"💥 Error sending event end email notification: {str(e)}")

@router.post("/camera-config")
async def set_camera_config(config: CameraConfig):
    """Save camera configuration from the frontend"""
    camera_config["internal_camera_id"] = config.internal_camera_id
    camera_config["external_camera_id"] = config.external_camera_id
    
    # Update video recording service
    video_recording_service.set_external_camera_config(config.external_camera_id)
    
    logger.info(f"Camera configuration updated: Internal: {config.internal_camera_id}, External: {config.external_camera_id}")
    
    return {"status": "success", "message": "Camera configuration saved successfully"}

@router.get("/external-camera-status")
async def get_external_camera_status():
    """Return the current status of recording and bee information"""
    return {
        "is_recording": bee_state["event_active"],
        "stream_url": None,
        "last_bee_status": bee_state["current_status"],
        "internal_camera_id": camera_config["internal_camera_id"],
        "external_camera_id": camera_config["external_camera_id"],
        "video_recording_status": video_recording_service.get_status()
    }

@router.get("/debug/conversion-status")
async def get_conversion_status():
    """Return video conversion service status"""
    return video_format_converter.get_conversion_status()

@router.get("/debug/bee-tracking-status")
async def get_bee_tracking_debug_status():
    """Return detailed debug information about bee tracking system"""
    return {
        "current_status": bee_state["current_status"],
        "status_sequence": bee_state["status_sequence"][-10:],
        "consecutive_detections": bee_state["consecutive_detections"],
        "event_active": bee_state["event_active"],
        "current_event_id": bee_state["current_event_id"],
        "position_history_count": len(bee_state["position_history"]),
        "configuration": {
            "center_line_x": CENTER_LINE_X,
            "frame_width": FRAME_WIDTH,
            "frame_height": FRAME_HEIGHT,
            "min_consecutive_detections": MIN_CONSECUTIVE_DETECTIONS
        },
        "camera_config": camera_config,
        "debug_info": {
            "last_statuses": bee_state["status_sequence"][-5:] if bee_state["status_sequence"] else [],
            "full_sequence": bee_state["status_sequence"],
            "looking_for_crossing": "inside → outside" if not bee_state["event_active"] else "outside → inside",
            "crossing_description": "Waiting for bee to exit hive" if not bee_state["event_active"] else "Waiting for bee to return to hive"
        },
        "video_recording": video_recording_service.get_status(),
        "video_conversion": video_format_converter.get_conversion_status()
    }

@router.get("/debug/model-info")
async def get_model_info():
    """Return information about the unified detection model and available classes"""
    if model_detect is None:
        return {"error": "Model not loaded", "models_loaded": False}
    
    try:
        class_names = model_detect.names if hasattr(model_detect, 'names') else "Not available"
        
        return {
            "models_loaded": True,
            "unified_model": {
                "model_file": "best.pt",
                "class_names": class_names,
                "available_classes": list(class_names.values()) if isinstance(class_names, dict) else class_names,
                "description": "Unified detection and classification model",
                "expected_classes": {
                    "0": "regular_bee",
                    "1": "marked_bee"
                }
            },
            "center_line_x": CENTER_LINE_X,
            "frame_dimensions": f"{FRAME_WIDTH}x{FRAME_HEIGHT}",
            "detection_threshold": 0.5,
            "position_history_size": POSITION_HISTORY_SIZE,
            "marked_bee_class_id": 1  # We only track class 1 (marked_bee)
        }
    except Exception as e:
        return {"error": str(e), "models_loaded": False}

@router.websocket("/live-stream")
async def live_stream(websocket: WebSocket):
    await websocket.accept()
    logger.info("🔌 WebSocket connection established")
    
    # Reset tracking state on new connection
    bee_tracking_service.reset_state()
    logger.info("🔄 Tracking state reset for new connection")
    
    try:
        while True:
            # Receive frame data
            data = await websocket.receive_bytes()
            
            # Convert bytes to numpy array
            nparr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                logger.warning("Invalid frame received")
                continue
            
            # Process frame with current settings
            processed_frame, bee_status, current_time, bee_x, bee_y = process_frame(frame)
            
            # Add frame to video recording service
            video_recording_service.add_processed_frame(processed_frame)
            
            # Handle bee tracking if status detected
            if bee_status and bee_x is not None and bee_y is not None:
                # Use the actual bee coordinates for event detection
                status, event_action = detect_sequence_pattern(bee_x, bee_y, current_time)
                
                if event_action:
                    await handle_bee_event(event_action, bee_status, current_time, processed_frame)
            
            # Encode and send processed frame
            _, buffer = cv2.imencode('.jpg', processed_frame, 
                                   [cv2.IMWRITE_JPEG_QUALITY, 80])
            
            # Send status update
            status_update = {
                "bee_status": bee_status,
                "current_time": current_time.isoformat(),
                "external_camera_status": {
                    "is_recording": video_recording_service.is_event_recording,
                    "current_event_id": video_recording_service.current_event_id
                },
                "settings_applied": {
                    "detection_enabled": current_settings.processing.detection_enabled,
                    "drawing_enabled": any([
                        current_settings.processing.draw_bee_path,
                        current_settings.processing.draw_center_line,
                        current_settings.processing.draw_roi_box
                    ])
                }
            }
            
            await websocket.send_text(json.dumps(status_update))
            await websocket.send_bytes(buffer.tobytes())
            
    except WebSocketDisconnect:
        logger.info("🔌 WebSocket disconnected")
        
        # Clean up videos if enabled
        if current_settings.processing.auto_delete_videos_after_session:
            try:
                await cleanup_session_videos()
                logger.info("🗑️ Session videos cleaned up as per settings")
            except Exception as e:
                logger.error(f"Failed to cleanup session videos: {e}")
    
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        logger.info("🔌 WebSocket connection closed")

async def cleanup_session_videos():
    """Clean up videos from current session if auto-delete is enabled"""
    if not current_settings.processing.auto_delete_videos_after_session:
        return
    
    try:
        import shutil
        import glob
        
        # Clean up temp videos
        temp_videos = glob.glob("/data/videos/temp_videos/*.mp4")
        for video_path in temp_videos:
            try:
                os.remove(video_path)
                logger.info(f"Deleted temp video: {video_path}")
            except Exception as e:
                logger.error(f"Failed to delete {video_path}: {e}")
        
        # Clean up recent event videos if not keeping originals
        if not current_settings.processing.keep_original_videos:
            # Only delete videos from the last hour to avoid deleting important recordings
            import time
            current_time = time.time()
            one_hour_ago = current_time - 3600
            
            event_videos = glob.glob("/data/videos/events/*/internal_camera_*.mp4")
            event_videos.extend(glob.glob("/data/videos/events/*/external_camera_*.mp4"))
            
            for video_path in event_videos:
                try:
                    file_time = os.path.getmtime(video_path)
                    if file_time > one_hour_ago:  # Only delete recent files
                        os.remove(video_path)
                        logger.info(f"Deleted recent event video: {video_path}")
                except Exception as e:
                    logger.error(f"Failed to delete {video_path}: {e}")
        
        logger.info("✅ Session video cleanup completed")
        
    except Exception as e:
        logger.error(f"Error during video cleanup: {e}")

@router.post("/settings")
async def update_settings(settings: SystemSettings):
    """Update system settings"""
    global current_settings
    try:
        current_settings = settings
        
        # Update bee tracking service configuration
        bee_tracking_service.state["consecutive_detections"] = {
            "inside": 0, "outside": 0
        }
        
        logger.info("✅ System settings updated successfully")
        logger.info(f"Detection enabled: {settings.processing.detection_enabled}")
        logger.info(f"Drawing settings: path={settings.processing.draw_bee_path}, center_line={settings.processing.draw_center_line}")
        logger.info(f"Video settings: auto_delete={settings.processing.auto_delete_videos_after_session}")
        
        return {
            "status": "success",
            "message": "Settings updated successfully",
            "settings": current_settings.dict()
        }
    except Exception as e:
        logger.error(f"Failed to update settings: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update settings: {str(e)}")

@router.get("/settings")
async def get_settings():
    """Get current system settings"""
    return {
        "status": "success",
        "settings": current_settings.dict()
    }

@router.post("/settings/reset")
async def reset_settings():
    """Reset settings to default values"""
    global current_settings
    try:
        current_settings = SystemSettings()
        logger.info("🔄 Settings reset to defaults")
        return {
            "status": "success",
            "message": "Settings reset to defaults",
            "settings": current_settings.dict()
        }
    except Exception as e:
        logger.error(f"Failed to reset settings: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to reset settings: {str(e)}")

@router.get("/settings/presets")
async def get_settings_presets():
    """Get predefined settings presets"""
    presets = {
        "default": SystemSettings().dict(),
        "minimal_processing": SystemSettings(
            processing=ProcessingSettings(
                draw_bee_path=False,
                draw_center_line=False,
                draw_roi_box=False,
                draw_confidence_scores=False,
                draw_timestamp=False,
                computer_vision_fallback=False
            )
        ).dict(),
        "maximum_debugging": SystemSettings(
            processing=ProcessingSettings(
                draw_bee_path=True,
                draw_center_line=True,
                draw_roi_box=True,
                draw_status_text=True,
                draw_confidence_scores=True,
                draw_timestamp=True,
                draw_frame_counter=True,
                computer_vision_fallback=True
            )
        ).dict(),
        "production_optimized": SystemSettings(
            processing=ProcessingSettings(
                draw_bee_path=True,
                draw_center_line=True,
                draw_roi_box=False,
                draw_status_text=False,
                draw_confidence_scores=False,
                draw_timestamp=False,
                auto_delete_videos_after_session=True,
                auto_convert_videos=True
            )
        ).dict()
    }
    
    return {
        "status": "success",
        "presets": presets
    }

@router.post("/settings/preset/{preset_name}")
async def apply_settings_preset(preset_name: str):
    """Apply a predefined settings preset"""
    global current_settings
    
    try:
        presets_response = await get_settings_presets()
        presets = presets_response["presets"]
        
        if preset_name not in presets:
            raise HTTPException(status_code=404, detail=f"Preset '{preset_name}' not found")
        
        preset_data = presets[preset_name]
        current_settings = SystemSettings(**preset_data)
        
        logger.info(f"✅ Applied settings preset: {preset_name}")
        
        return {
            "status": "success",
            "message": f"Applied preset: {preset_name}",
            "settings": current_settings.dict()
        }
        
    except Exception as e:
        logger.error(f"Failed to apply preset {preset_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to apply preset: {str(e)}")

@router.post("/debug/set-initial-status")
async def set_initial_bee_status(status_data: dict = Body(...)):
    """Set initial bee status for testing"""
    status = status_data.get("status")
    
    if status not in ["inside", "outside"]:
        return {"error": "Status must be 'inside' or 'outside'"}
    
    try:
        bee_state["current_status"] = status
        bee_state["status_sequence"] = [status]
        bee_state["consecutive_detections"] = {"inside": 0, "outside": 0}
        
        logger.info(f"🎯 Manual status set: {status}")
        
        return {
            "success": True,
            "message": f"Initial bee status set to {status}",
            "current_state": {
                "current_status": bee_state["current_status"],
                "last_status": status
            }
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

@router.post("/debug/reset-tracking")
async def reset_tracking_manually():
    """Manually reset the bee tracking state"""
    global bee_state
    bee_state = {
        "current_status": None,
        "status_sequence": [],
        "consecutive_detections": {"inside": 0, "outside": 0},
        "event_active": False,
        "current_event_id": None,
        "position_history": []
    }
    logger.info("🔄 Bee tracking state reset")
    return {"message": "Bee tracking state has been reset", "success": True}

@router.post("/test-conversion")
async def test_video_conversion():
    """Test video conversion functionality"""
    try:
        # Get conversion status
        status = video_format_converter.get_conversion_status()
        
        # Test with a sample video if available
        test_results = {
            "ffmpeg_available": status["ffmpeg_available"],
            "conversion_directory": status["converted_directory"],
            "test_conversion": "No sample video available for testing"
        }
        
        return test_results
        
    except Exception as e:
        logger.error(f"Video conversion test error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/test-email")
async def test_email():
    """Test email functionality"""
    try:
        success = email_service.test_email_connection()
        if success:
            test_success = email_service.send_bee_detection_notification(
                event_type="exit",
                timestamp=datetime.now(),
                additional_info={"test": True}
            )
            return {
                "connection_test": success,
                "send_test": test_success,
                "message": "Email test completed successfully" if test_success else "Email connection OK but send failed"
            }
        else:
            return {
                "connection_test": success,
                "send_test": False,
                "message": "Email connection failed"
            }
    except Exception as e:
        logger.error(f"Email test error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/videos/{file_path:path}")
async def serve_video(file_path: str, request: Request):
    """Serve video files with proper streaming support"""
    try:
        range_header = request.headers.get('range')
        
        if range_header:
            return video_service.get_streaming_response(file_path, range_header)
        else:
            return video_service.get_file_response(file_path)
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error serving video {file_path}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/manual-convert/{event_id}")
async def manual_convert_event_videos(event_id: str):
    """Manually trigger video conversion for a specific event"""
    try:
        from app.services.service import get_event_by_id
        
        # Get event from database
        event = await get_event_by_id(event_id)
        if not event:
            raise HTTPException(status_code=404, detail="Event not found")
        
        # Check if conversion is already completed
        if event.conversion_status == "completed":
            return {
                "message": "Event videos already converted",
                "event_id": event_id,
                "internal_converted": event.internal_video_url_converted,
                "external_converted": event.external_video_url_converted
            }
        
        # Check if conversion is already in progress
        if event.conversion_status == "processing":
            return {
                "message": "Conversion already in progress",
                "event_id": event_id,
                "status": "processing"
            }
        
        # Get original video paths
        internal_path = None
        external_path = None
        
        if event.internal_video_url:
            # Convert URL to full path
            internal_path = os.path.join(VIDEOS_DIR, event.internal_video_url.lstrip("/videos/"))
        
        if event.external_video_url:
            # Convert URL to full path
            external_path = os.path.join(VIDEOS_DIR, event.external_video_url.lstrip("/videos/"))
        
        if not internal_path and not external_path:
            raise HTTPException(status_code=400, detail="No videos found for this event")
        
        # Start conversion in background
        logger.info(f"🎬 Manual conversion triggered for event {event_id}")
        conversion_thread = threading.Thread(
            target=lambda: asyncio.run(video_conversion_callback(event_id, internal_path, external_path)),
            daemon=True
        )
        conversion_thread.start()
        
        return {
            "message": "Video conversion started",
            "event_id": event_id,
            "internal_path": internal_path,
            "external_path": external_path
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Manual conversion error for event {event_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/conversion-status/{event_id}")
async def get_event_conversion_status(event_id: str):
    """Get conversion status for a specific event"""
    try:
        from app.services.service import get_event_by_id
        
        event = await get_event_by_id(event_id)
        if not event:
            raise HTTPException(status_code=404, detail="Event not found")
        
        return {
            "event_id": event_id,
            "conversion_status": event.conversion_status,
            "conversion_error": event.conversion_error,
            "original_videos": {
                "internal": event.internal_video_url,
                "external": event.external_video_url
            },
            "converted_videos": {
                "internal": event.internal_video_url_converted,
                "external": event.external_video_url_converted
            },
            "time_out": event.time_out.isoformat() if event.time_out else None,
            "time_in": event.time_in.isoformat() if event.time_in else None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting conversion status for event {event_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/events-needing-conversion")
async def get_events_needing_conversion():
    """Get list of events that need video conversion"""
    try:
        all_events = await get_all_events()
        
        events_needing_conversion = []
        for event in all_events:
            # Check if event has videos but no conversion status or failed conversion
            has_videos = bool(event.internal_video_url or event.external_video_url)
            needs_conversion = (
                has_videos and 
                (not event.conversion_status or event.conversion_status in ["pending", "failed"])
            )
            
            if needs_conversion:
                events_needing_conversion.append({
                    "id": event.id,
                    "time_out": event.time_out.isoformat() if event.time_out else None,
                    "time_in": event.time_in.isoformat() if event.time_in else None,
                    "conversion_status": event.conversion_status,
                    "conversion_error": event.conversion_error,
                    "has_internal_video": bool(event.internal_video_url),
                    "has_external_video": bool(event.external_video_url),
                    "internal_video_converted": bool(event.internal_video_url_converted),
                    "external_video_converted": bool(event.external_video_url_converted)
                })
        
        return {
            "total_events": len(all_events),
            "events_needing_conversion": len(events_needing_conversion),
            "events": events_needing_conversion
        }
        
    except Exception as e:
        logger.error(f"Error getting events needing conversion: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/batch-convert-events")
async def batch_convert_events():
    """Convert all events that need conversion"""
    try:
        all_events = await get_all_events()
        
        conversion_tasks = []
        for event in all_events:
            # Check if event needs conversion
            has_videos = bool(event.internal_video_url or event.external_video_url)
            needs_conversion = (
                has_videos and 
                (not event.conversion_status or event.conversion_status in ["pending", "failed"]) and
                event.conversion_status != "processing"
            )
            
            if needs_conversion:
                internal_path = None
                external_path = None
                
                if event.internal_video_url:
                    internal_path = os.path.join(VIDEOS_DIR, event.internal_video_url.lstrip("/videos/"))
                
                if event.external_video_url:
                    external_path = os.path.join(VIDEOS_DIR, event.external_video_url.lstrip("/videos/"))
                
                conversion_tasks.append({
                    "event_id": event.id,
                    "internal_path": internal_path,
                    "external_path": external_path
                })
        
        # Start conversion for all events
        for task in conversion_tasks:
            logger.info(f"🎬 Starting batch conversion for event {task['event_id']}")
            conversion_thread = threading.Thread(
                target=lambda t=task: asyncio.run(video_conversion_callback(
                    t["event_id"], t["internal_path"], t["external_path"]
                )),
                daemon=True
            )
            conversion_thread.start()
            
            # Small delay between conversions to avoid overwhelming the system
            import time
            time.sleep(1)
        
        return {
            "message": f"Batch conversion started for {len(conversion_tasks)} events",
            "events_to_convert": len(conversion_tasks),
            "conversion_tasks": [{"event_id": t["event_id"]} for t in conversion_tasks]
        }
        
    except Exception as e:
        logger.error(f"Batch conversion error: {e}")
        raise HTTPException(status_code=500, detail=str(e)) 