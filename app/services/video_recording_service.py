"""
Video Recording Service - ניהול הקלטת וידאו עם buffer
"""
import cv2
import time
import os
import threading
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any
import numpy as np
from collections import deque

logger = logging.getLogger(__name__)

class VideoRecordingService:
    """שירות להקלטת וידאו עם תמיכה ב-buffer לשמירת 5 שניות אחורה"""
    
    def __init__(self, videos_dir: str = "/data/videos"):
        self.videos_dir = videos_dir
        self.buffer_seconds = 5
        self.fps = 20
        self.buffer_size = self.buffer_seconds * self.fps  # 100 frames for 5 seconds
        
        # Internal camera buffer (always recording last 5 seconds)
        self.internal_buffer = deque(maxlen=self.buffer_size)
        self.internal_buffer_lock = threading.Lock()
        
        # Recording state
        self.is_event_recording = False
        self.current_event_id = None
        self.internal_writer = None
        self.external_writer = None
        self.internal_output_path = None
        self.external_output_path = None
        
        # External camera state
        self.external_camera_config = None
        self.external_capture = None
        self.external_thread = None
        self.stop_external = False
        
        # Ensure directories exist
        os.makedirs(f"{videos_dir}/events", exist_ok=True)
        os.makedirs(f"{videos_dir}/temp", exist_ok=True)
        
        logger.info("🎥 Video recording service initialized")
    
    def add_frame_to_buffer(self, frame: np.ndarray):
        """הוסף פריים ל-buffer הפנימי (קורה כל הזמן)"""
        with self.internal_buffer_lock:
            timestamp = time.time()
            # Store frame with timestamp
            self.internal_buffer.append((frame.copy(), timestamp))
    
    def set_external_camera_config(self, camera_id: str):
        """הגדר מצלמה חיצונית"""
        self.external_camera_config = camera_id
        logger.info(f"External camera configured: {camera_id}")
    
    def start_event_recording(self, event_id: str) -> Dict[str, str]:
        """התחל הקלטת אירוע - שמור buffer + התחל הקלטה חדשה"""
        if self.is_event_recording:
            logger.warning("Event recording already active")
            return {}
        
        self.is_event_recording = True
        self.current_event_id = event_id
        
        timestamp = int(time.time())
        event_dir = f"{self.videos_dir}/events/event_{event_id}"
        os.makedirs(event_dir, exist_ok=True)
        
        # Paths for the two videos
        self.internal_output_path = f"{event_dir}/internal_camera_{timestamp}.mp4"
        self.external_output_path = f"{event_dir}/external_camera_{timestamp}.mp4"
        
        # 1. Start internal camera recording (with buffer)
        self._start_internal_recording()
        
        # 2. Start external camera recording
        self._start_external_recording()
        
        logger.info(f"🎬 Event recording started for event {event_id}")
        logger.info(f"   Internal: {self.internal_output_path}")
        logger.info(f"   External: {self.external_output_path}")
        
        return {
            "internal_video": self.internal_output_path,
            "external_video": self.external_output_path
        }
    
    def stop_event_recording(self) -> Dict[str, str]:
        """עצור הקלטת אירוע ושמור קבצים"""
        if not self.is_event_recording:
            logger.warning("No event recording active")
            return {}
        
        # Stop internal recording
        if self.internal_writer:
            self.internal_writer.release()
            self.internal_writer = None
        
        # Stop external recording
        self._stop_external_recording()
        
        # Get the file paths before resetting
        internal_path = self.internal_output_path
        external_path = self.external_output_path
        
        # Reset state
        self.is_event_recording = False
        self.current_event_id = None
        self.internal_output_path = None
        self.external_output_path = None
        
        logger.info(f"🛑 Event recording stopped")
        
        # Return relative paths for the database
        return {
            "internal_video": f"/videos/{os.path.relpath(internal_path, self.videos_dir)}" if internal_path and os.path.exists(internal_path) else None,
            "external_video": f"/videos/{os.path.relpath(external_path, self.videos_dir)}" if external_path and os.path.exists(external_path) else None
        }
    
    def _start_internal_recording(self):
        """התחל הקלטה פנימית עם שמירת ה-buffer"""
        try:
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            
            # Use the last frame dimensions from buffer
            with self.internal_buffer_lock:
                if len(self.internal_buffer) > 0:
                    sample_frame = self.internal_buffer[-1][0]
                    height, width = sample_frame.shape[:2]
                else:
                    # Default dimensions
                    width, height = 640, 480
            
            self.internal_writer = cv2.VideoWriter(
                self.internal_output_path, fourcc, self.fps, (width, height)
            )
            
            if not self.internal_writer.isOpened():
                raise Exception("Failed to open internal video writer")
            
            # Write buffered frames (last 5 seconds)
            with self.internal_buffer_lock:
                buffer_frames = list(self.internal_buffer)
            
            logger.info(f"Writing {len(buffer_frames)} buffered frames to internal video")
            for frame, timestamp in buffer_frames:
                self.internal_writer.write(frame)
            
            logger.info("✅ Internal recording started with buffer")
            
        except Exception as e:
            logger.error(f"Failed to start internal recording: {e}")
            if self.internal_writer:
                self.internal_writer.release()
                self.internal_writer = None
    
    def _start_external_recording(self):
        """התחל הקלטה חיצונית"""
        if not self.external_camera_config:
            logger.warning("No external camera configured")
            return
        
        try:
            # Create video writer for external camera
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            width, height = 640, 480  # Default for external camera
            
            self.external_writer = cv2.VideoWriter(
                self.external_output_path, fourcc, self.fps, (width, height)
            )
            
            if not self.external_writer.isOpened():
                raise Exception("Failed to open external video writer")
            
            # Start external camera thread
            self.stop_external = False
            self.external_thread = threading.Thread(target=self._external_recording_thread)
            self.external_thread.daemon = True
            self.external_thread.start()
            
            logger.info("✅ External recording started")
            
        except Exception as e:
            logger.error(f"Failed to start external recording: {e}")
            if self.external_writer:
                self.external_writer.release()
                self.external_writer = None
    
    def _external_recording_thread(self):
        """Thread להקלטה מהמצלמה החיצונית"""
        try:
            # Try to open external camera
            if self.external_camera_config.isdigit():
                camera_id = int(self.external_camera_config)
            else:
                camera_id = self.external_camera_config
            
            self.external_capture = cv2.VideoCapture(camera_id)
            
            if not self.external_capture.isOpened():
                logger.error(f"Failed to open external camera: {camera_id}")
                # Create mock frames instead
                self._create_mock_external_video()
                return
            
            logger.info(f"External camera opened: {camera_id}")
            frame_count = 0
            start_time = time.time()
            
            while not self.stop_external and self.external_writer:
                ret, frame = self.external_capture.read()
                
                if not ret:
                    logger.warning("Failed to read from external camera")
                    break
                
                # Add timestamp overlay
                elapsed = time.time() - start_time
                timestamp_text = f"External Camera - {elapsed:.2f}s"
                cv2.putText(frame, timestamp_text, (20, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                
                # Write frame
                self.external_writer.write(frame)
                frame_count += 1
                
                # Control frame rate
                time.sleep(1.0 / self.fps)
            
            logger.info(f"External recording finished. Frames: {frame_count}")
            
        except Exception as e:
            logger.error(f"Error in external recording thread: {e}")
        finally:
            if self.external_capture:
                self.external_capture.release()
                self.external_capture = None
    
    def _create_mock_external_video(self):
        """צור וידאו mock למצלמה חיצונית"""
        if not self.external_writer:
            return
        
        logger.info("Creating mock external video")
        start_time = time.time()
        frame_count = 0
        
        while not self.stop_external and self.external_writer:
            # Create mock frame
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            
            elapsed = time.time() - start_time
            timestamp_text = f"Mock External Camera - {elapsed:.2f}s"
            cv2.putText(frame, timestamp_text, (20, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            
            camera_text = f"Camera: {self.external_camera_config} (MOCK)"
            cv2.putText(frame, camera_text, (20, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
            cv2.putText(frame, "MOCK RECORDING", (20, 450),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            self.external_writer.write(frame)
            frame_count += 1
            
            time.sleep(1.0 / self.fps)
        
        logger.info(f"Mock external recording finished. Frames: {frame_count}")
    
    def _stop_external_recording(self):
        """עצור הקלטה חיצונית"""
        self.stop_external = True
        
        if self.external_thread:
            self.external_thread.join(timeout=5.0)
            self.external_thread = None
        
        if self.external_writer:
            self.external_writer.release()
            self.external_writer = None
        
        if self.external_capture:
            self.external_capture.release()
            self.external_capture = None
        
        logger.info("🛑 External recording stopped")
    
    def add_processed_frame(self, frame: np.ndarray):
        """הוסף פריים מעובד להקלטה (אם פעילה)"""
        # Always add to buffer
        self.add_frame_to_buffer(frame)
        
        # If event recording is active, also write to internal video
        if self.is_event_recording and self.internal_writer:
            self.internal_writer.write(frame)
    
    def get_status(self) -> Dict[str, Any]:
        """קבל סטטוס נוכחי של ההקלטה"""
        return {
            "is_event_recording": self.is_event_recording,
            "current_event_id": self.current_event_id,
            "buffer_frames": len(self.internal_buffer),
            "internal_video": self.internal_output_path,
            "external_video": self.external_output_path,
            "external_camera_configured": bool(self.external_camera_config)
        }

# Create singleton instance
video_recording_service = VideoRecordingService() 