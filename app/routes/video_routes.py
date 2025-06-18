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
import threading
from pydantic import BaseModel
import logging
import asyncio

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CameraConfig(BaseModel):
    internal_camera_id: str
    external_camera_id: str

router = APIRouter()

# Initialize YOLO model - unified detection & classification
model_detect = YOLO("best.pt")  # Your trained model that detects regular_bee (0) and marked_bee (1)
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
    """Check if point is on the inside (right) side of the hive
    Right side = inside hive, Left side = outside hive"""
    return x_center > CENTER_LINE_X

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
    if consecutive_inside >= MIN_CONSECUTIVE_DETECTIONS:
        confirmed_status = "inside"
    elif consecutive_outside >= MIN_CONSECUTIVE_DETECTIONS:
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
    """Draw vertical center line separating inside/outside areas"""
    height, width = frame.shape[:2]
    
    # Update global frame dimensions
    global FRAME_WIDTH, FRAME_HEIGHT, CENTER_LINE_X
    FRAME_WIDTH = width
    FRAME_HEIGHT = height  
    CENTER_LINE_X = width // 2
    
    # Draw main center line (bright yellow)
    cv2.line(frame, (CENTER_LINE_X, 0), (CENTER_LINE_X, height), (0, 255, 255), 3)
    
    # Draw buffer zones (thin lines)
    buffer_width = 20
    cv2.line(frame, (CENTER_LINE_X - buffer_width, 0), (CENTER_LINE_X - buffer_width, height), (128, 128, 255), 1)
    cv2.line(frame, (CENTER_LINE_X + buffer_width, 0), (CENTER_LINE_X + buffer_width, height), (128, 128, 255), 1)
    
    # Add area labels
    cv2.putText(frame, "OUTSIDE", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 165, 0), 2)
    cv2.putText(frame, "INSIDE HIVE", (CENTER_LINE_X + 50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    
    # Add center line label
    cv2.putText(frame, "HIVE ENTRANCE", (CENTER_LINE_X - 80, height - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    return frame

def draw_bee_path(frame):
    """Draw the bee's movement path on the frame with colors based on inside/outside status"""
    position_history = bee_state["position_history"]
    
    if len(position_history) < 2:
        return frame
    
    # Draw path lines connecting consecutive points
    for i in range(1, len(position_history)):
        prev_x, prev_y, prev_time, prev_status = position_history[i-1]
        curr_x, curr_y, curr_time, curr_status = position_history[i]
        
        # Color based on current status: Green for inside, Orange for outside
        color = (0, 255, 0) if curr_status == "inside" else (0, 165, 255)  # Green inside, Orange outside
        
        # Draw line between consecutive points
        cv2.line(frame, (int(prev_x), int(prev_y)), (int(curr_x), int(curr_y)), color, 2)
        
        # Draw small circles at each point
        cv2.circle(frame, (int(curr_x), int(curr_y)), 3, color, -1)
    
    # Draw larger circle at the most recent position
    if position_history:
        last_x, last_y, _, last_status = position_history[-1]
        color = (0, 255, 0) if last_status == "inside" else (0, 165, 255)  # Green inside, Orange outside
        cv2.circle(frame, (int(last_x), int(last_y)), 8, color, 2)
        
        # Add status text near the bee
        status_text = f"BEE: {last_status.upper()}"
        cv2.putText(frame, status_text, (int(last_x) + 15, int(last_y) - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    return frame

def find_bee_location_in_frame(frame):
    """
    Try to estimate bee location in frame using computer vision techniques
    This is a temporary solution for classification models that don't provide bounding boxes
    """
    try:
        # Convert to different color spaces for better detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Method 1: Look for dark spots (bees are usually darker)
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Find dark regions (potential bees) - use adaptive threshold for better results
        binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Filter contours by size (bee-like objects)
            min_area = 50  # Minimum area for a bee
            max_area = 2000  # Maximum area for a bee
            
            logger.debug(f"🔍 Found {len(contours)} contours, filtering by size ({min_area}-{max_area})")
            
            bee_candidates = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if min_area <= area <= max_area:
                    # Get contour center
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        
                        # Calculate aspect ratio
                        x, y, w, h = cv2.boundingRect(contour)
                        aspect_ratio = float(w) / h
                        
                        # Bees are roughly oval, so aspect ratio should be reasonable
                        if 0.3 <= aspect_ratio <= 3.0:
                            bee_candidates.append((cx, cy, area, aspect_ratio))
                            logger.debug(f"  Candidate: pos=({cx}, {cy}), area={area}, aspect_ratio={aspect_ratio:.2f}")
            
            if bee_candidates:
                # Sort by area (larger objects more likely to be bees)
                bee_candidates.sort(key=lambda x: x[2], reverse=True)
                
                # Return the center of the largest candidate
                best_candidate = bee_candidates[0]
                logger.debug(f"🎯 Selected best candidate: pos=({best_candidate[0]}, {best_candidate[1]}), area={best_candidate[2]}, aspect_ratio={best_candidate[3]:.2f}")
                return (best_candidate[0], best_candidate[1])
            else:
                logger.debug("🔍 No suitable contour candidates found (size/aspect ratio filters)")
        else:
            logger.debug("🔍 No contours found in binary image")
        
        # Method 2: Look for colored markers if the bee has a colored mark
        # Define color ranges for common bee markers (red, blue, yellow, etc.)
        color_ranges = [
            # Red marker
            ([0, 50, 50], [10, 255, 255]),
            ([170, 50, 50], [180, 255, 255]),
            # Blue marker  
            ([100, 50, 50], [130, 255, 255]),
            # Yellow marker
            ([20, 50, 50], [30, 255, 255]),
            # Green marker
            ([40, 50, 50], [80, 255, 255])
        ]
        
        for lower, upper in color_ranges:
            lower = np.array(lower, dtype=np.uint8)
            upper = np.array(upper, dtype=np.uint8)
            
            # Create mask for this color range
            mask = cv2.inRange(hsv, lower, upper)
            
            # Find contours in the mask
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if 20 <= area <= 500:  # Smaller area for colored markers
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        logger.debug(f"🎨 Found colored marker at ({cx}, {cy}) with area {area}")
                        return (cx, cy)
        
        logger.debug("🔍 No bee candidates found using computer vision")
        return None
        
    except Exception as e:
        logger.error(f"Error in find_bee_location_in_frame: {e}")
        return None

def process_frame(frame):
    """Process frame for bee detection and classification"""
    time_str = format_time(time.time() * 1000)
    current_time = datetime.now()
    processed_frame = frame.copy()

    try:
        results_detect = model_detect(processed_frame, conf=0.4)  # Lower confidence threshold to ensure boxes are created
    except Exception as e:
        logger.error(f"YOLO detection failed: {e}")
        cv2.putText(processed_frame, f"Detection Error: {str(e)[:50]}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return processed_frame, None, current_time, None
    
    # Debug logging to understand the results structure
    logger.info(f"🔍 YOLO Results Debug:")
    logger.info(f"   results_detect type: {type(results_detect)}")
    logger.info(f"   results_detect length: {len(results_detect) if results_detect else 0}")
    
    if results_detect and len(results_detect) > 0:
        result = results_detect[0]
        logger.info(f"   First result type: {type(result)}")
        logger.info(f"   Has 'boxes' attribute: {hasattr(result, 'boxes')}")
        
        if hasattr(result, 'boxes'):
            boxes = result.boxes
            logger.info(f"   boxes type: {type(boxes)}")
            logger.info(f"   boxes is None: {boxes is None}")
            
            if boxes is not None:
                logger.info(f"   Has 'xyxy' attribute: {hasattr(boxes, 'xyxy')}")
                logger.info(f"   Has 'cls' attribute: {hasattr(boxes, 'cls')}")
                logger.info(f"   Has 'conf' attribute: {hasattr(boxes, 'conf')}")
                
                if hasattr(boxes, 'xyxy'):
                    logger.info(f"   xyxy shape: {boxes.xyxy.shape if hasattr(boxes.xyxy, 'shape') else 'No shape attr'}")
                    logger.info(f"   Number of detections: {len(boxes.xyxy) if boxes.xyxy is not None else 0}")

    bee_detected = False
    bee_status = None
    event_action = None
    

    
    if results_detect and len(results_detect) > 0:
        result = results_detect[0]
        
        # Try to access boxes first (standard format)
        if hasattr(result, 'boxes') and result.boxes is not None:
            boxes = result.boxes
            
            # Check all required attributes exist and are not None
            has_xyxy = hasattr(boxes, 'xyxy') and boxes.xyxy is not None
            has_cls = hasattr(boxes, 'cls') and boxes.cls is not None  
            has_conf = hasattr(boxes, 'conf') and boxes.conf is not None
            
            if has_xyxy and has_cls and has_conf:
                # Check if we have actual detections
                try:
                    num_detections = len(boxes.xyxy)
                    if num_detections > 0:
                        logger.info(f"✅ Standard YOLO detection format found with {num_detections} detections")
                        
                        # Process each detection
                        for i in range(num_detections):
                            try:
                                # Get bounding box coordinates
                                xyxy_box = boxes.xyxy[i]
                                if len(xyxy_box) >= 4:
                                    x1, y1, x2, y2 = map(int, xyxy_box[:4])
                                    
                                    # Get class information from detection results
                                    cls_id = int(boxes.cls[i])  # Class ID from detection
                                    conf = float(boxes.conf[i])  # Confidence from detection
                                    cls_name = result.names.get(cls_id, f"class_{cls_id}")  # Class name from model
                                    
                                    logger.debug(f"Detection {i}: class='{cls_name}' (ID: {cls_id}), confidence={conf:.3f}")
                                    
                                    # Check if this is a marked bee (class 1) with sufficient confidence
                                    is_marked_bee = (cls_id == 1 and conf > 0.3)  # Temporarily lowered threshold from 0.5 to 0.3 for testing
                                    
                                    if is_marked_bee:
                                        bee_detected = True
                                        logger.info(f"✅ MARKED BEE DETECTED: class='{cls_name}', confidence={conf:.3f}, box=[{x1},{y1},{x2},{y2}]")
                                        
                                        x_center = (x1 + x2) // 2
                                        y_center = (y1 + y2) // 2
                                        
                                        # Check if enough consecutive detections
                                        consecutive_inside = bee_state["consecutive_detections"]["inside"]
                                        consecutive_outside = bee_state["consecutive_detections"]["outside"]
                                        
                                        if consecutive_inside < MIN_CONSECUTIVE_DETECTIONS and consecutive_outside < MIN_CONSECUTIVE_DETECTIONS:
                                            logger.debug(f"Not enough consecutive detections: inside={consecutive_inside}, outside={consecutive_outside}")
                                        
                                        bee_status, event_action = detect_sequence_pattern(x_center, y_center, current_time)
                                        
                                        # Add to position history
                                        raw_status = "inside" if is_inside_hive(x_center, y_center) else "outside"
                                        timestamp = current_time.timestamp()
                                        bee_state["position_history"].append((x_center, y_center, timestamp, raw_status))
                                        if len(bee_state["position_history"]) > POSITION_HISTORY_SIZE:
                                            bee_state["position_history"] = bee_state["position_history"][-POSITION_HISTORY_SIZE:]
                                        
                                        logger.info(f"Bee position: ({x_center}, {y_center}), Status: {bee_status}, Raw: {raw_status}, Event: {event_action}")
                                        logger.info(f"Center line check: inside={is_inside_hive(x_center, y_center)}, Center_X={CENTER_LINE_X}")
                                        logger.info(f"Consecutive counts: inside={consecutive_inside}, outside={consecutive_outside}")
                                        logger.info(f"Sequence: {' → '.join(bee_state['status_sequence'][-5:]) if bee_state['status_sequence'] else 'Empty'}")
                                        
                                        # Store bee image for email notifications
                                        padding = 30
                                        x1_padded = max(0, x1 - padding)
                                        y1_padded = max(0, y1 - padding) 
                                        x2_padded = min(processed_frame.shape[1], x2 + padding)
                                        y2_padded = min(processed_frame.shape[0], y2 + padding)
                                        
                                        bee_image = processed_frame[y1_padded:y2_padded, x1_padded:x2_padded].copy()
                                        
                                        if not hasattr(process_frame, 'last_bee_image'):
                                            process_frame.last_bee_image = None
                                        process_frame.last_bee_image = bee_image
                                        
                                        # Draw bee visualization
                                        color = (0, 255, 0) if raw_status == "inside" else (255, 165, 0)
                                        cv2.rectangle(processed_frame, (x1, y1), (x2, y2), color, 3)
                                        
                                        # Draw crosshair at center
                                        cv2.line(processed_frame, (x_center - 10, y_center), (x_center + 10, y_center), (255, 255, 255), 2)
                                        cv2.line(processed_frame, (x_center, y_center - 10), (x_center, y_center + 10), (255, 255, 255), 2)
                                        
                                        # Draw info
                                        info_texts = [
                                            f"MARKED BEE - {raw_status.upper()}",
                                            f"Pos: ({x_center}, {y_center})",
                                            f"Conf: {conf:.2f}",
                                            f"Event: {'ACTIVE' if bee_state['event_active'] else 'INACTIVE'}",
                                            f"Cons: I={consecutive_inside} O={consecutive_outside}"
                                        ]
                                        
                                        y_offset = max(50, y1 - 120)
                                        for j, text in enumerate(info_texts):
                                            cv2.putText(processed_frame, text, (x1, y_offset + j * 20),
                                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                                            cv2.putText(processed_frame, text, (x1, y_offset + j * 20),
                                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                                        
                                        break  # Only process first marked bee detection
                                        
                            except Exception as e:
                                logger.error(f"Error processing detection {i}: {e}")
                                continue
                    else:
                        logger.debug("No detections found in boxes.xyxy")
                        
                except Exception as e:
                    logger.error(f"Error accessing detection data: {e}")
            else:
                logger.warning("⚠️ Boxes object missing required attributes (xyxy, cls, or conf)")
                if not has_xyxy:
                    logger.warning("  - Missing or None xyxy attribute")
                if not has_cls:
                    logger.warning("  - Missing or None cls attribute") 
                if not has_conf:
                    logger.warning("  - Missing or None conf attribute")
        
        # If boxes is None or no detections, try to access raw prediction data
        if not bee_detected:
            logger.info("🔄 Boxes is None or empty, trying to access raw prediction data...")
            
            # Try to access probs for classification-style results
            if hasattr(result, 'probs') and result.probs is not None:
                logger.info("📊 Found probs attribute - processing classification results")
                try:
                    probs = result.probs
                    if hasattr(probs, 'data') and probs.data is not None:
                        # Get class probabilities
                        class_probs = probs.data.cpu().numpy() if hasattr(probs.data, 'cpu') else probs.data
                        
                        # Find marked_bee class (class 1)
                        if len(class_probs) > 1:
                            marked_bee_conf = class_probs[1]  # Class 1 = marked_bee
                            logger.info(f"📊 Classification result: marked_bee confidence = {marked_bee_conf:.3f}")
                            
                            if marked_bee_conf > 0.3:  # Lower threshold for classification
                                logger.info(f"📊 Classification detected marked_bee: confidence={marked_bee_conf:.3f}")
                                
                                # Try to find actual bee location using computer vision
                                frame_height, frame_width = processed_frame.shape[:2]
                                bee_location = find_bee_location_in_frame(processed_frame)
                                
                                if bee_location is not None:
                                    # Only mark as detected if we actually found a bee in the frame
                                    bee_detected = True
                                    x_center, y_center = bee_location
                                    logger.info(f"✅ MARKED BEE DETECTED via classification + CV: confidence={marked_bee_conf:.3f}, location=({x_center}, {y_center})")
                                    
                                    # Create a bounding box around estimated location
                                    box_size = 100
                                    x1 = max(0, x_center - box_size // 2)
                                    y1 = max(0, y_center - box_size // 2)
                                    x2 = min(frame_width, x_center + box_size // 2)
                                    y2 = min(frame_height, y_center + box_size // 2)
                                    
                                    # Process the detection only if we found a real location
                                    
                                    # Process the detection
                                    consecutive_inside = bee_state["consecutive_detections"]["inside"]
                                    consecutive_outside = bee_state["consecutive_detections"]["outside"]
                                    
                                    bee_status, event_action = detect_sequence_pattern(x_center, y_center, current_time)
                                    
                                    # Add to position history
                                    raw_status = "inside" if is_inside_hive(x_center, y_center) else "outside"
                                    timestamp = current_time.timestamp()
                                    bee_state["position_history"].append((x_center, y_center, timestamp, raw_status))
                                    if len(bee_state["position_history"]) > POSITION_HISTORY_SIZE:
                                        bee_state["position_history"] = bee_state["position_history"][-POSITION_HISTORY_SIZE:]
                                    
                                    logger.info(f"Bee position (classification): ({x_center}, {y_center}), Status: {bee_status}, Raw: {raw_status}, Event: {event_action}")
                                    
                                    # Draw visualization for classification result
                                    color = (0, 255, 0) if raw_status == "inside" else (255, 165, 0)
                                    cv2.rectangle(processed_frame, (x1, y1), (x2, y2), color, 3)
                                    
                                    # Draw crosshair at center
                                    cv2.line(processed_frame, (x_center - 10, y_center), (x_center + 10, y_center), (255, 255, 255), 2)
                                    cv2.line(processed_frame, (x_center, y_center - 10), (x_center, y_center + 10), (255, 255, 255), 2)
                                    
                                    # Draw info
                                    info_texts = [
                                        f"MARKED BEE (CLASSIFICATION) - {raw_status.upper()}",
                                        f"Pos: ({x_center}, {y_center})",
                                        f"Conf: {marked_bee_conf:.2f}",
                                        f"Event: {'ACTIVE' if bee_state['event_active'] else 'INACTIVE'}",
                                        f"Cons: I={consecutive_inside} O={consecutive_outside}"
                                    ]
                                    
                                    y_offset = 50
                                    for j, text in enumerate(info_texts):
                                        cv2.putText(processed_frame, text, (x1, y_offset + j * 20),
                                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                                        cv2.putText(processed_frame, text, (x1, y_offset + j * 20),
                                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                                else:
                                    # Classification says there's a bee, but CV couldn't find it
                                    # Don't mark as detected - this prevents false center-screen detections
                                    logger.debug(f"⚠️ Classification detected marked_bee (conf={marked_bee_conf:.3f}) but CV couldn't locate it in frame - no detection shown")
                        
                except Exception as e:
                    logger.error(f"Error processing classification results: {e}")
            
            # Try alternative formats if still no detection
            if not bee_detected:
                if hasattr(result, 'pred') and result.pred is not None:
                    logger.info("🔄 Trying alternative format (pred attribute)")
                    # Alternative format handling can be added here if needed
                elif hasattr(result, 'pandas'):
                    logger.info("🔄 Trying pandas format")
                    # Pandas format handling
                    try:
                        df = result.pandas().xyxy[0]
                        marked_bees = df[df['name'] == 'marked_bee']
                        if len(marked_bees) > 0:
                            logger.info(f"Found {len(marked_bees)} marked bees via pandas format")
                            # Process pandas format detections here
                    except Exception as e:
                        logger.error(f"Pandas format processing failed: {e}")
                else:
                    logger.debug("No alternative detection formats available")
    else:
        logger.debug("No detection results or results format unexpected")


    # Always draw center line and bee path
    processed_frame = draw_center_line_info(processed_frame)
    processed_frame = draw_bee_path(processed_frame)

    # Add enhanced status information to the frame
    last_status = bee_state["status_sequence"][-1] if bee_state["status_sequence"] else None
    
    status_info = [
        f"Current Status: {bee_state['current_status'] or 'No bee detected'}",
        f"Last Status: {last_status or 'None'}",
        f"Tracking Points: {len(bee_state['position_history'])}",
        f"Event: {'ACTIVE' if bee_state['event_active'] else 'INACTIVE'}",
        f"Sequence: {' → '.join(bee_state['status_sequence'][-5:]) if bee_state['status_sequence'] else 'Empty'}"
    ]
    
    # Draw status box in top-left corner
    y_offset = 30
    for i, text in enumerate(status_info):
        cv2.putText(processed_frame, text, (10, y_offset + i * 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(processed_frame, text, (10, y_offset + i * 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    
    # Add timestamp to frame
    cv2.putText(processed_frame, time_str, (10, processed_frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(processed_frame, time_str, (10, processed_frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
    
    # If no bee detected, gradually reset counters
    if not bee_detected:
        bee_state["consecutive_detections"]["inside"] = max(0, bee_state["consecutive_detections"]["inside"] - 1)
        bee_state["consecutive_detections"]["outside"] = max(0, bee_state["consecutive_detections"]["outside"] - 1)

    return processed_frame, bee_status, current_time, event_action

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
        
        # Send email notification
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
    logger.info("WebSocket connected")
    
    # Reset bee tracking state on new connection
    logger.info("🔄 Resetting bee tracking state for new connection")
    bee_state["current_status"] = None
    bee_state["status_sequence"] = []
    bee_state["consecutive_detections"] = {"inside": 0, "outside": 0}
    bee_state["event_active"] = False
    bee_state["current_event_id"] = None
    bee_state["position_history"] = []
    logger.info("✅ Bee tracking state reset complete")

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
                logger.warning("Invalid frame received")
                continue

            processed_frame, bee_status, current_time, event_action = process_frame(frame)
            
            # Add processed frame to video recording service buffer
            video_recording_service.add_processed_frame(processed_frame)
            
            if bee_status is not None:
                try:
                    status_update = {
                        "bee_status": bee_status,
                        "external_camera_status": bee_state["event_active"],
                        "event_action": event_action,
                        "position_history_count": len(bee_state["position_history"]),
                        "consecutive_inside": bee_state["consecutive_detections"]["inside"],
                        "consecutive_outside": bee_state["consecutive_detections"]["outside"],
                        "event_active": bee_state["event_active"],
                        "status_sequence": bee_state["status_sequence"][-5:]
                    }
                    await websocket.send_text(json.dumps(status_update))
                except Exception as e:
                    logger.error(f"Error sending status update: {e}")
            
            if event_action:
                try:
                    bee_image = getattr(process_frame, 'last_bee_image', None)
                    await handle_bee_event(event_action, bee_status, current_time, bee_image)
                    logger.info(f"Processed event action: {event_action}")
                except Exception as e:
                    logger.error(f"Error in bee event handling: {e}")

            if video_writer is None:
                height, width = processed_frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                fps = 20.0 
                video_writer = cv2.VideoWriter(processed_filename, fourcc, fps, (width, height))
                if not video_writer.isOpened():
                    raise Exception("Failed to open VideoWriter")

            video_writer.write(processed_frame)
            frame_count += 1

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"Error: {e}")
    finally:
        if video_writer:
            video_writer.release()
            logger.info(f"Saved processed video: {processed_filename}")
            
            current_event_id = bee_state["current_event_id"]
            if current_event_id:
                video_url = video_service.save_video_locally(processed_filename)
                if video_url:
                    event_update = EventUpdate(video_url=video_url)
                    await update_event(current_event_id, event_update)

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