"""
Camera Synchronizer - Handles frame-level timestamp synchronization between cameras
"""
import asyncio
import numpy as np
from collections import deque
from datetime import datetime, timedelta
from typing import Tuple, Optional, Dict, Any
from threading import RLock
from dataclasses import dataclass
import logging
import time

logger = logging.getLogger(__name__)

@dataclass
class FrameData:
    """Represents a timestamped frame"""
    frame: np.ndarray
    timestamp: datetime
    server_timestamp: datetime
    frame_id: int

class CameraSynchronizer:
    """Handles synchronization between internal and external camera frames"""
    
    def __init__(self, session_id: str, max_buffer_size: int = 30):
        self.session_id = session_id
        self.max_buffer_size = max_buffer_size
        
        # Frame buffers for both cameras
        self.internal_frame_buffer = deque(maxlen=max_buffer_size)
        self.external_frame_buffer = deque(maxlen=max_buffer_size)
        
        # Synchronization state
        self.sync_baseline: Optional[datetime] = None
        self.drift_compensation: float = 0.0
        self.frame_counter = 0
        self.sync_lock = RLock()
        
        # Synchronization metrics
        self.sync_metrics = {
            "frames_synchronized": 0,
            "average_drift": 0.0,
            "max_drift": 0.0,
            "last_sync_time": None,
            "buffer_health": {
                "internal_buffer_size": 0,
                "external_buffer_size": 0,
                "buffer_balance": 0  # difference between buffer sizes
            }
        }
        
        # Configuration
        self.max_drift_threshold = 2000  # milliseconds
        self.sync_window = 50  # milliseconds - frames within this window are considered synchronized
        
        logger.info(f"🎬 CameraSynchronizer initialized for session {session_id}")
    
    def establish_sync_baseline(self) -> datetime:
        """Establish synchronization baseline timestamp"""
        with self.sync_lock:
            self.sync_baseline = datetime.now()
            self.drift_compensation = 0.0
            logger.info(f"⏰ Sync baseline established for session {self.session_id} at {self.sync_baseline}")
            return self.sync_baseline
    
    def add_internal_frame(self, frame: np.ndarray, timestamp: datetime) -> bool:
        """Add frame from internal camera to buffer"""
        try:
            with self.sync_lock:
                self.frame_counter += 1
                server_timestamp = datetime.now()
                
                frame_data = FrameData(
                    frame=frame,
                    timestamp=timestamp,
                    server_timestamp=server_timestamp,
                    frame_id=self.frame_counter
                )
                
                self.internal_frame_buffer.append(frame_data)
                self._update_buffer_metrics()
                
                logger.debug(f"📹 Internal frame {self.frame_counter} added to session {self.session_id}")
                return True
                
        except Exception as e:
            logger.error(f"❌ Error adding internal frame to session {self.session_id}: {e}")
            return False
    
    def add_external_frame(self, frame: np.ndarray, timestamp: datetime) -> bool:
        """Add frame from external camera to buffer"""
        try:
            with self.sync_lock:
                self.frame_counter += 1
                server_timestamp = datetime.now()
                
                frame_data = FrameData(
                    frame=frame,
                    timestamp=timestamp,
                    server_timestamp=server_timestamp,
                    frame_id=self.frame_counter
                )
                
                self.external_frame_buffer.append(frame_data)
                self._update_buffer_metrics()
                
                logger.debug(f"📹 External frame {self.frame_counter} added to session {self.session_id}")
                return True
                
        except Exception as e:
            logger.error(f"❌ Error adding external frame to session {self.session_id}: {e}")
            return False
    
    def get_synchronized_frames(self) -> Optional[Tuple[FrameData, FrameData]]:
        """Get synchronized frame pair based on timestamps"""
        with self.sync_lock:
            if not self.internal_frame_buffer or not self.external_frame_buffer:
                return None
            
            # Find best matching frames within sync window
            best_match = None
            min_time_diff = float('inf')
            
            # Search through recent frames for best timestamp match
            for internal_frame in list(self.internal_frame_buffer)[-10:]:  # Check last 10 frames
                for external_frame in list(self.external_frame_buffer)[-10:]:
                    
                    # Calculate time difference
                    time_diff = abs((internal_frame.timestamp - external_frame.timestamp).total_seconds() * 1000)
                    
                    # Check if within sync window and better than current best
                    if time_diff <= self.sync_window and time_diff < min_time_diff:
                        min_time_diff = time_diff
                        best_match = (internal_frame, external_frame)
            
            if best_match:
                self.sync_metrics["frames_synchronized"] += 1
                self.sync_metrics["last_sync_time"] = datetime.now()
                
                # Update drift metrics
                self._update_drift_metrics(min_time_diff)
                
                logger.debug(f"🎯 Synchronized frames for session {self.session_id} with {min_time_diff:.1f}ms difference")
                return best_match
            
            return None
    
    def calculate_drift(self) -> float:
        """Calculate current timing drift between cameras"""
        with self.sync_lock:
            if len(self.internal_frame_buffer) < 2 or len(self.external_frame_buffer) < 2:
                return 0.0
            
            # Calculate average time difference over recent frames
            recent_diffs = []
            
            # Compare timestamps of recent frames
            internal_recent = list(self.internal_frame_buffer)[-5:]
            external_recent = list(self.external_frame_buffer)[-5:]
            
            for i_frame in internal_recent:
                for e_frame in external_recent:
                    time_diff = (i_frame.timestamp - e_frame.timestamp).total_seconds() * 1000
                    recent_diffs.append(time_diff)
            
            if recent_diffs:
                average_drift = sum(recent_diffs) / len(recent_diffs)
                self.drift_compensation = average_drift
                return average_drift
            
            return 0.0
    
    def adjust_timing(self, timestamp: datetime) -> datetime:
        """Adjust timestamp based on calculated drift compensation"""
        if self.drift_compensation != 0.0:
            adjustment = timedelta(milliseconds=self.drift_compensation)
            return timestamp + adjustment
        return timestamp
    
    def get_sync_status(self) -> Dict[str, Any]:
        """Get current synchronization status and metrics"""
        with self.sync_lock:
            current_drift = self.calculate_drift()
            
            return {
                "session_id": self.session_id,
                "sync_baseline": self.sync_baseline.isoformat() if self.sync_baseline else None,
                "current_drift_ms": current_drift,
                "drift_compensation_ms": self.drift_compensation,
                "sync_window_ms": self.sync_window,
                "max_drift_threshold_ms": self.max_drift_threshold,
                "sync_metrics": self.sync_metrics.copy(),
                "buffer_status": {
                    "internal_buffer_size": len(self.internal_frame_buffer),
                    "external_buffer_size": len(self.external_frame_buffer),
                    "max_buffer_size": self.max_buffer_size,
                    "buffer_utilization": {
                        "internal": len(self.internal_frame_buffer) / self.max_buffer_size,
                        "external": len(self.external_frame_buffer) / self.max_buffer_size
                    }
                },
                "sync_health": {
                    "is_synchronized": abs(current_drift) <= self.max_drift_threshold,
                    "drift_status": "good" if abs(current_drift) <= 50 else "warning" if abs(current_drift) <= 100 else "critical",
                    "buffer_balance": abs(len(self.internal_frame_buffer) - len(self.external_frame_buffer))
                }
            }
    
    def reset_synchronization(self):
        """Reset synchronization state"""
        with self.sync_lock:
            self.internal_frame_buffer.clear()
            self.external_frame_buffer.clear()
            self.sync_baseline = None
            self.drift_compensation = 0.0
            self.frame_counter = 0
            
            # Reset metrics
            self.sync_metrics = {
                "frames_synchronized": 0,
                "average_drift": 0.0,
                "max_drift": 0.0,
                "last_sync_time": None,
                "buffer_health": {
                    "internal_buffer_size": 0,
                    "external_buffer_size": 0,
                    "buffer_balance": 0
                }
            }
            
            logger.info(f"🔄 Synchronization reset for session {self.session_id}")
    
    def _update_buffer_metrics(self):
        """Update buffer health metrics"""
        self.sync_metrics["buffer_health"] = {
            "internal_buffer_size": len(self.internal_frame_buffer),
            "external_buffer_size": len(self.external_frame_buffer),
            "buffer_balance": abs(len(self.internal_frame_buffer) - len(self.external_frame_buffer))
        }
    
    def _update_drift_metrics(self, time_diff: float):
        """Update drift calculation metrics"""
        # Update average drift (running average)
        current_avg = self.sync_metrics["average_drift"]
        sync_count = self.sync_metrics["frames_synchronized"]
        
        if sync_count == 1:
            self.sync_metrics["average_drift"] = time_diff
        else:
            # Running average calculation
            self.sync_metrics["average_drift"] = ((current_avg * (sync_count - 1)) + time_diff) / sync_count
        
        # Update max drift
        if time_diff > self.sync_metrics["max_drift"]:
            self.sync_metrics["max_drift"] = time_diff
    
    def is_sync_healthy(self) -> bool:
        """Check if synchronization is healthy"""
        with self.sync_lock:
            current_drift = abs(self.calculate_drift())
            buffer_balance = abs(len(self.internal_frame_buffer) - len(self.external_frame_buffer))
            
            # Sync is healthy if:
            # 1. Drift is within threshold
            # 2. Buffers are reasonably balanced
            # 3. Both buffers have recent frames
            return (current_drift <= self.max_drift_threshold and
                    buffer_balance <= 5 and
                    len(self.internal_frame_buffer) > 0 and
                    len(self.external_frame_buffer) > 0)
    
    def get_buffer_status(self) -> Dict[str, Any]:
        """Get detailed buffer status"""
        with self.sync_lock:
            return {
                "internal_buffer": {
                    "size": len(self.internal_frame_buffer),
                    "max_size": self.max_buffer_size,
                    "utilization": len(self.internal_frame_buffer) / self.max_buffer_size,
                    "oldest_frame_age": (datetime.now() - self.internal_frame_buffer[0].server_timestamp).total_seconds() 
                                       if self.internal_frame_buffer else None,
                    "newest_frame_age": (datetime.now() - self.internal_frame_buffer[-1].server_timestamp).total_seconds() 
                                       if self.internal_frame_buffer else None
                },
                "external_buffer": {
                    "size": len(self.external_frame_buffer),
                    "max_size": self.max_buffer_size,
                    "utilization": len(self.external_frame_buffer) / self.max_buffer_size,
                    "oldest_frame_age": (datetime.now() - self.external_frame_buffer[0].server_timestamp).total_seconds() 
                                       if self.external_frame_buffer else None,
                    "newest_frame_age": (datetime.now() - self.external_frame_buffer[-1].server_timestamp).total_seconds() 
                                       if self.external_frame_buffer else None
                },
                "balance": abs(len(self.internal_frame_buffer) - len(self.external_frame_buffer))
            } 