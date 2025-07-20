"""
Dual Camera WebSocket Handler - Manages both internal and external camera streams
"""
import asyncio
import cv2
import numpy as np
import json
from datetime import datetime
from typing import Optional, Dict, Any
from fastapi import WebSocket
from starlette.websockets import WebSocketDisconnect
import logging

from app.services.camera_session_manager import camera_session_manager
from app.services.websocket_connection_manager import websocket_connection_manager
from app.services.camera_synchronizer import CameraSynchronizer

logger = logging.getLogger(__name__)

class DualCameraWebSocketHandler:
    """Handles WebSocket connections for dual camera sessions"""
    
    def __init__(self):
        self.active_synchronizers: Dict[str, CameraSynchronizer] = {}
        self.processing_stats: Dict[str, Dict] = {}
        logger.info("🎬 DualCameraWebSocketHandler initialized")
    
    async def handle_internal_camera_connection(self, websocket: WebSocket, session_id: str):
        """Handle internal camera WebSocket connection"""
        client_ip = websocket.client.host if websocket.client else "unknown"
        logger.info(f"🔌 Internal camera connection attempt from {client_ip} for session {session_id}")
        
        await websocket.accept()
        logger.info(f"✅ Internal camera WebSocket accepted for session {session_id}")
        
        try:
            # Register connection
            success = await websocket_connection_manager.register_connection(session_id, "internal", websocket)
            if not success:
                await websocket.close(code=4000, reason="Failed to register session")
                return
            
            # Perform session handshake
            handshake_success = await websocket_connection_manager.perform_session_handshake(session_id, "internal", websocket)
            if not handshake_success:
                await websocket.close(code=4001, reason="Handshake failed")
                return
            
            # Initialize synchronizer if this is the first camera
            if session_id not in self.active_synchronizers:
                self.active_synchronizers[session_id] = CameraSynchronizer(session_id)
                self.processing_stats[session_id] = {
                    "internal_frames_received": 0,
                    "external_frames_received": 0,
                    "synchronized_frames": 0,
                    "session_start": datetime.now(),
                    "last_activity": datetime.now()
                }
            
            # Start frame processing loop
            await self._process_internal_camera_frames(session_id, websocket)
            
        except WebSocketDisconnect:
            logger.info(f"🔌 Internal camera disconnected from session {session_id}")
        except Exception as e:
            logger.error(f"💥 Internal camera error for session {session_id}: {e}")
        finally:
            await websocket_connection_manager.unregister_connection(session_id, "internal")
            await self._cleanup_session_if_empty(session_id)
    
    async def handle_external_camera_connection(self, websocket: WebSocket, session_id: str):
        """Handle external camera WebSocket connection"""
        client_ip = websocket.client.host if websocket.client else "unknown"
        logger.info(f"🔌 External camera connection attempt from {client_ip} for session {session_id}")
        
        await websocket.accept()
        logger.info(f"✅ External camera WebSocket accepted for session {session_id}")
        
        try:
            # Register connection
            success = await websocket_connection_manager.register_connection(session_id, "external", websocket)
            if not success:
                await websocket.close(code=4000, reason="Failed to register session")
                return
            
            # Perform session handshake
            handshake_success = await websocket_connection_manager.perform_session_handshake(session_id, "external", websocket)
            if not handshake_success:
                await websocket.close(code=4001, reason="Handshake failed")
                return
            
            # Initialize synchronizer if needed
            if session_id not in self.active_synchronizers:
                self.active_synchronizers[session_id] = CameraSynchronizer(session_id)
                self.processing_stats[session_id] = {
                    "internal_frames_received": 0,
                    "external_frames_received": 0,
                    "synchronized_frames": 0,
                    "session_start": datetime.now(),
                    "last_activity": datetime.now()
                }
            
            # Start frame processing loop
            await self._process_external_camera_frames(session_id, websocket)
            
        except WebSocketDisconnect:
            logger.info(f"🔌 External camera disconnected from session {session_id}")
        except Exception as e:
            logger.error(f"💥 External camera error for session {session_id}: {e}")
        finally:
            await websocket_connection_manager.unregister_connection(session_id, "external")
            await self._cleanup_session_if_empty(session_id)
    
    async def _process_internal_camera_frames(self, session_id: str, websocket: WebSocket):
        """Process frames from internal camera"""
        logger.info(f"📹 Started internal camera frame processing for session {session_id}")
        
        try:
            while True:
                # Receive frame data
                data = await websocket.receive_bytes()
                current_time = datetime.now()
                
                # Convert bytes to numpy array
                nparr = np.frombuffer(data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is None:
                    logger.warning(f"⚠️ Invalid internal frame received for session {session_id}")
                    continue
                
                # Update statistics
                self.processing_stats[session_id]["internal_frames_received"] += 1
                self.processing_stats[session_id]["last_activity"] = current_time
                
                # Add frame to synchronizer
                synchronizer = self.active_synchronizers.get(session_id)
                if synchronizer:
                    synchronizer.add_internal_frame(frame, current_time)
                
                # Process frame (bee detection, etc.)
                processed_frame = await self._process_frame(session_id, frame, "internal", current_time)
                
                # Send processed frame back to client
                if processed_frame is not None:
                    _, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    await websocket.send_bytes(buffer.tobytes())
                
                # Send status update
                status_update = await self._get_session_status_update(session_id, "internal")
                await websocket.send_text(json.dumps(status_update, ensure_ascii=False))
                
        except WebSocketDisconnect:
            pass  # Handled in calling function
        except Exception as e:
            logger.error(f"💥 Error processing internal frames for session {session_id}: {e}")
    
    async def _process_external_camera_frames(self, session_id: str, websocket: WebSocket):
        """Process frames from external camera"""
        logger.info(f"📹 Started external camera frame processing for session {session_id}")
        
        try:
            while True:
                # Receive frame data
                data = await websocket.receive_bytes()
                current_time = datetime.now()
                
                # Convert bytes to numpy array
                nparr = np.frombuffer(data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is None:
                    logger.warning(f"⚠️ Invalid external frame received for session {session_id}")
                    continue
                
                # Update statistics
                self.processing_stats[session_id]["external_frames_received"] += 1
                self.processing_stats[session_id]["last_activity"] = current_time
                
                # Add frame to synchronizer
                synchronizer = self.active_synchronizers.get(session_id)
                if synchronizer:
                    synchronizer.add_external_frame(frame, current_time)
                
                # Process frame (simple recording, no detection for external)
                processed_frame = await self._process_frame(session_id, frame, "external", current_time)
                
                # Send processed frame back to client
                if processed_frame is not None:
                    _, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    await websocket.send_bytes(buffer.tobytes())
                
                # Send status update
                status_update = await self._get_session_status_update(session_id, "external")
                await websocket.send_text(json.dumps(status_update, ensure_ascii=False))
                
        except WebSocketDisconnect:
            pass  # Handled in calling function
        except Exception as e:
            logger.error(f"💥 Error processing external frames for session {session_id}: {e}")
    
    async def _process_frame(self, session_id: str, frame: np.ndarray, camera_type: str, timestamp: datetime) -> Optional[np.ndarray]:
        """Process individual frame based on camera type"""
        try:
            processed_frame = frame.copy()
            
            # Add timestamp overlay
            timestamp_text = timestamp.strftime("%H:%M:%S.%f")[:-3]
            cv2.putText(processed_frame, f"{camera_type.upper()}: {timestamp_text}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Add session ID overlay
            cv2.putText(processed_frame, f"Session: {session_id}", 
                       (10, processed_frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            if camera_type == "internal":
                # Internal camera: Add bee detection processing
                processed_frame = await self._process_internal_frame(session_id, processed_frame, timestamp)
            else:
                # External camera: Add recording indicator
                session = camera_session_manager.get_session(session_id)
                if session and session.recording_active:
                    cv2.putText(processed_frame, "● REC", 
                               (processed_frame.shape[1] - 100, 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            # Add synchronization status
            synchronizer = self.active_synchronizers.get(session_id)
            if synchronizer:
                sync_status = synchronizer.get_sync_status()
                sync_text = f"Sync: {sync_status['sync_health']['drift_status']}"
                cv2.putText(processed_frame, sync_text, 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            return processed_frame
            
        except Exception as e:
            logger.error(f"💥 Error processing {camera_type} frame for session {session_id}: {e}")
            return frame
    
    async def _process_internal_frame(self, session_id: str, frame: np.ndarray, timestamp: datetime) -> np.ndarray:
        """Process internal camera frame with bee detection"""
        try:
            # Import here to avoid circular imports
            from app.services.bee_tracking_service import bee_tracking_service
            from app.routes.video_routes import process_frame, detect_sequence_pattern
            
            # Get or create session tracker
            tracker = bee_tracking_service.get_session_tracker(session_id)
            if not tracker:
                tracker = bee_tracking_service.create_session_tracker(session_id)
            
            # Process frame for bee detection (reuse existing logic)
            processed_frame, bee_status, _, bee_x, bee_y = process_frame(frame)
            
            # Handle bee tracking if status detected
            if bee_status and bee_x is not None and bee_y is not None:
                status, event_action = detect_sequence_pattern(bee_x, bee_y, timestamp)
                
                if event_action:
                    # Trigger event coordination
                    await self._coordinate_event(session_id, event_action, bee_status, timestamp)
            
            return processed_frame
            
        except Exception as e:
            logger.error(f"💥 Error in internal frame processing for session {session_id}: {e}")
            return frame
    
    async def _coordinate_event(self, session_id: str, event_action: str, bee_status: str, timestamp: datetime):
        """Coordinate event between cameras"""
        try:
            logger.info(f"🎯 Event coordination: {event_action} for session {session_id}")
            
            # Update session recording status
            session = camera_session_manager.get_session(session_id)
            if not session:
                return
            
            if event_action == "start_event":
                session.recording_active = True
                session.current_event_id = f"event_{timestamp.strftime('%Y%m%d_%H%M%S')}"
                
                # Notify both cameras about recording start
                await websocket_connection_manager.broadcast_to_session(session_id, {
                    "type": "recording_started",
                    "event_id": session.current_event_id,
                    "timestamp": timestamp.isoformat(),
                    "bee_status": bee_status
                })
                
            elif event_action == "end_event":
                session.recording_active = False
                event_id = session.current_event_id
                session.current_event_id = None
                
                # Notify both cameras about recording end
                await websocket_connection_manager.broadcast_to_session(session_id, {
                    "type": "recording_stopped",
                    "event_id": event_id,
                    "timestamp": timestamp.isoformat(),
                    "bee_status": bee_status
                })
            
            logger.info(f"✅ Event coordination completed for session {session_id}")
            
        except Exception as e:
            logger.error(f"💥 Error coordinating event for session {session_id}: {e}")
    
    async def _get_session_status_update(self, session_id: str, camera_type: str) -> Dict[str, Any]:
        """Get status update for session"""
        try:
            session = camera_session_manager.get_session(session_id)
            synchronizer = self.active_synchronizers.get(session_id)
            stats = self.processing_stats.get(session_id, {})
            
            status = {
                "type": "status_update",
                "session_id": session_id,
                "camera_type": camera_type,
                "timestamp": datetime.now().isoformat(),
                "session_status": session.get_status() if session else None,
                "sync_status": synchronizer.get_sync_status() if synchronizer else None,
                "processing_stats": stats,
                "connection_status": websocket_connection_manager.get_connection_status(session_id)
            }
            
            return status
            
        except Exception as e:
            logger.error(f"💥 Error getting status update for session {session_id}: {e}")
            return {
                "type": "status_update",
                "session_id": session_id,
                "camera_type": camera_type,
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }
    
    async def _cleanup_session_if_empty(self, session_id: str):
        """Clean up session if no more connections"""
        try:
            connection_status = websocket_connection_manager.get_connection_status(session_id)
            
            if connection_status["total_connections"] == 0:
                logger.info(f"🧹 Cleaning up empty session {session_id}")
                
                # Clean up synchronizer
                if session_id in self.active_synchronizers:
                    del self.active_synchronizers[session_id]
                
                # Clean up stats
                if session_id in self.processing_stats:
                    del self.processing_stats[session_id]
                
                # Clean up session
                camera_session_manager.cleanup_session(session_id)
                
                logger.info(f"✅ Session {session_id} cleaned up successfully")
                
        except Exception as e:
            logger.error(f"💥 Error cleaning up session {session_id}: {e}")
    
    def get_session_processing_stats(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get processing statistics for session"""
        stats = self.processing_stats.get(session_id)
        if not stats:
            return None
        
        synchronizer = self.active_synchronizers.get(session_id)
        sync_stats = synchronizer.get_sync_status() if synchronizer else None
        
        return {
            "session_id": session_id,
            "processing_stats": stats,
            "sync_stats": sync_stats,
            "active_since": stats.get("session_start", "unknown"),
            "last_activity": stats.get("last_activity", "unknown")
        }
    
    def get_all_sessions_stats(self) -> Dict[str, Any]:
        """Get statistics for all active sessions"""
        all_stats = {}
        
        for session_id in self.processing_stats:
            all_stats[session_id] = self.get_session_processing_stats(session_id)
        
        return {
            "total_sessions": len(all_stats),
            "sessions": all_stats,
            "handler_status": "active" if all_stats else "idle"
        }

# Global singleton instance
dual_camera_websocket_handler = DualCameraWebSocketHandler() 