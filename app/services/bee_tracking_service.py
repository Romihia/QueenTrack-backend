"""
Bee Tracking Service - ניהול מעקב אחר דבורים
"""
import cv2
import time
import logging
from datetime import datetime
from typing import Tuple, Optional, Dict, Any
import numpy as np

logger = logging.getLogger(__name__)

# Configuration
PATH_HISTORY_SIZE = 50
MIN_CONSECUTIVE_DETECTIONS = 3
TRANSITION_COOLDOWN = 2.0
SEQUENCE_PATTERN = ["outside", "inside", "outside"]
VIDEO_BUFFER_SECONDS = 5
VIDEO_BUFFER_FPS = 20
HIVE_ENTRANCE_ROI = [400, 0, 600, 700]

class BeeTrackingService:
    """שירות למעקב אחר דבורים וזיהוי אירועים"""
    
    def __init__(self):
        self.state = {
            "current_status": None,
            "status_sequence": [],
            "current_event_id": None,
            "position_history": [],
            "last_transition_time": None,
            "consecutive_detections": {"inside": 0, "outside": 0},
            "event_active": False,
            "video_buffers": {
                "internal": [],
                "external": []
            }
        }
        logger.info("🐝 Bee tracking service initialized")
    
    def is_inside_roi(self, x_center: int, y_center: int, roi: list = None) -> bool:
        """בדוק אם הנקודה בתוך ה-ROI"""
        if roi is None:
            roi = HIVE_ENTRANCE_ROI
            
        x_min, y_min, x_max, y_max = roi
        
        # Define buffer zones (10% of ROI width/height)
        buffer_x = (x_max - x_min) * 0.1
        buffer_y = (y_max - y_min) * 0.1
        
        # Check current position
        is_inside = (x_min <= x_center <= x_max) and (y_min <= y_center <= y_max)
        
        # If we're in the buffer zone, maintain current status to prevent oscillation
        in_buffer_zone = (
            (x_min - buffer_x <= x_center <= x_min + buffer_x) or
            (x_max - buffer_x <= x_center <= x_max + buffer_x)
        ) and (y_min <= y_center <= y_max)
        
        if in_buffer_zone and self.state["current_status"] is not None:
            # Maintain current state in buffer zone
            return self.state["current_status"] == "inside"
            
        return is_inside
    
    def add_position_to_history(self, x: int, y: int, status: str, current_time: datetime):
        """הוסף נקודת מיקום להיסטוריה"""
        timestamp = current_time.timestamp()
        self.state["position_history"].append((x, y, timestamp, status))
        
        # Keep only recent positions
        if len(self.state["position_history"]) > PATH_HISTORY_SIZE:
            self.state["position_history"] = self.state["position_history"][-PATH_HISTORY_SIZE:]
    
    def detect_sequence_pattern(self, x_center: int, y_center: int, current_time: datetime) -> Tuple[str, Optional[str]]:
        """
        זיהוי רצף תנועות ומציאת דפוסי אירועים
        מחזיר: (current_status, event_action)
        """
        # Check if bee is inside ROI
        is_currently_inside = self.is_inside_roi(x_center, y_center)
        current_status = "inside" if is_currently_inside else "outside"
        
        # Update consecutive detection counters
        if current_status == "inside":
            self.state["consecutive_detections"]["inside"] += 1
            self.state["consecutive_detections"]["outside"] = 0
        else:
            self.state["consecutive_detections"]["outside"] += 1
            self.state["consecutive_detections"]["inside"] = 0
        
        # Always update current_status to the most recent confirmed status
        if self.state["consecutive_detections"][current_status] >= 1:
            self.state["current_status"] = current_status
        
        # Check if we have enough consecutive detections to confirm a status change
        if self.state["consecutive_detections"][current_status] < MIN_CONSECUTIVE_DETECTIONS:
            logger.debug(f"Not enough consecutive detections: {self.state['consecutive_detections'][current_status]}/{MIN_CONSECUTIVE_DETECTIONS}")
            return current_status, None
        
        # Check cooldown period
        if self.state["last_transition_time"]:
            time_since_last_transition = current_time.timestamp() - self.state["last_transition_time"]
            if time_since_last_transition < TRANSITION_COOLDOWN:
                logger.debug(f"In cooldown period: {time_since_last_transition:.1f}s remaining")
                return current_status, None
        
        # Check if this is a new status (different from the last in sequence)
        last_status = self.state["status_sequence"][-1] if self.state["status_sequence"] else None
        if last_status == current_status:
            return current_status, None  # No change
        
        # Add new status to sequence
        self.state["status_sequence"].append(current_status)
        self.state["last_transition_time"] = current_time.timestamp()
        
        # Keep only recent sequence (last 10 transitions)
        if len(self.state["status_sequence"]) > 10:
            self.state["status_sequence"] = self.state["status_sequence"][-10:]
        
        logger.info(f"🔄 Status sequence updated: {' → '.join(self.state['status_sequence'][-5:])}")
        
        # Check for event patterns
        event_action = None
        
        if not self.state["event_active"]:
            # Check for event start pattern: outside → inside → outside
            if len(self.state["status_sequence"]) >= 3:
                recent_sequence = self.state["status_sequence"][-3:]
                if recent_sequence == SEQUENCE_PATTERN:
                    event_action = "start_event"
                    self.state["event_active"] = True
                    logger.warning(f"🚨 EVENT START PATTERN DETECTED: {' → '.join(recent_sequence)}")
        else:
            # Event is active, check for end pattern: outside → inside
            if len(self.state["status_sequence"]) >= 2:
                recent_transition = self.state["status_sequence"][-2:]
                if recent_transition == ["outside", "inside"]:
                    event_action = "end_event"
                    self.state["event_active"] = False
                    logger.warning(f"🏠 EVENT END PATTERN DETECTED: {' → '.join(recent_transition)}")
        
        return current_status, event_action
    
    def reset_state(self):
        """איפוס מצב המעקב"""
        self.state = {
            "current_status": None,
            "status_sequence": [],
            "current_event_id": None,
            "position_history": [],
            "last_transition_time": None,
            "consecutive_detections": {"inside": 0, "outside": 0},
            "event_active": False,
            "video_buffers": {
                "internal": [],
                "external": []
            }
        }
        logger.info("🔄 Bee tracking state reset")
    
    def set_initial_status(self, status: str):
        """הגדרת סטטוס התחלתי"""
        if status not in ["inside", "outside"]:
            raise ValueError("Status must be 'inside' or 'outside'")
        
        self.state["current_status"] = status
        self.state["status_sequence"] = [status]
        self.state["consecutive_detections"] = {"inside": 0, "outside": 0}
        self.state["last_transition_time"] = None
        
        logger.info(f"🎯 Manual status set: {status}")
    
    def get_debug_info(self) -> Dict[str, Any]:
        """קבלת מידע דיבוג מפורט"""
        current_time = time.time()
        last_transition_time = self.state.get("last_transition_time")
        time_since_last_transition = None
        if last_transition_time:
            time_since_last_transition = current_time - last_transition_time
        
        return {
            "bee_state": {
                "current_status": self.state["current_status"],
                "status_sequence": self.state["status_sequence"],
                "event_active": self.state["event_active"],
                "current_event_id": self.state["current_event_id"],
                "consecutive_detections": self.state["consecutive_detections"],
                "position_history_count": len(self.state["position_history"]),
                "last_transition_time": last_transition_time,
                "time_since_last_transition": time_since_last_transition
            },
            "pattern_matching": {
                "required_pattern": SEQUENCE_PATTERN,
                "current_sequence": self.state["status_sequence"][-5:],
                "pattern_matched": (self.state["status_sequence"][-3:] == SEQUENCE_PATTERN 
                                  if len(self.state["status_sequence"]) >= 3 else False)
            },
            "position_history": self.state["position_history"][-10:] if self.state["position_history"] else [],
            "configuration": {
                "roi": HIVE_ENTRANCE_ROI,
                "path_history_size": PATH_HISTORY_SIZE,
                "min_consecutive_detections": MIN_CONSECUTIVE_DETECTIONS,
                "transition_cooldown": TRANSITION_COOLDOWN,
                "sequence_pattern": SEQUENCE_PATTERN,
                "video_buffer_seconds": VIDEO_BUFFER_SECONDS
            }
        }

# Create singleton instance
bee_tracking_service = BeeTrackingService() 