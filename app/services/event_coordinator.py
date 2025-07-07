"""
Event Coordinator - Coordinates events between dual cameras and manages video recording
"""
import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
from threading import RLock
import uuid

from app.services.camera_session_manager import camera_session_manager
from app.services.websocket_connection_manager import websocket_connection_manager
from app.services.video_recording_service import video_recording_service
from app.services.email_service import email_service
from app.schemas.schema import EventCreate
from app.services.service import create_event

logger = logging.getLogger(__name__)

class EventCoordinator:
    """Coordinates events between dual camera sessions"""
    
    _instance = None
    _lock = RLock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, '_initialized'):
            self.active_events: Dict[str, Dict[str, Any]] = {}
            self.event_history: List[Dict[str, Any]] = []
            self.notification_queue: asyncio.Queue = asyncio.Queue()
            self.coordinator_lock = RLock()
            self._initialized = True
            logger.info("🎯 EventCoordinator initialized")
    
    async def start_event(self, session_id: str, trigger_data: Dict[str, Any]) -> Optional[str]:
        """Start a new event for the session"""
        try:
            with self.coordinator_lock:
                # Check if session already has an active event
                if session_id in self.active_events:
                    logger.warning(f"⚠️ Session {session_id} already has an active event")
                    return self.active_events[session_id]["event_id"]
                
                # Generate unique event ID
                event_id = f"event_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
                
                # Create event record
                event_data = {
                    "event_id": event_id,
                    "session_id": session_id,
                    "start_time": datetime.now(),
                    "trigger_data": trigger_data,
                    "status": "starting",
                    "cameras_recording": {},
                    "video_paths": {},
                    "notifications_sent": [],
                    "bee_status": trigger_data.get("bee_status", "unknown")
                }
                
                self.active_events[session_id] = event_data
                
                logger.info(f"🎬 Event {event_id} started for session {session_id}")
                
                # Start recording on both cameras
                recording_result = await self._start_dual_camera_recording(session_id, event_id)
                
                if recording_result:
                    event_data["cameras_recording"] = recording_result
                    event_data["status"] = "recording"
                    
                    # Send event start notification to cameras
                    await self._notify_cameras_event_start(session_id, event_id, trigger_data)
                    
                    # Queue email notification
                    await self._queue_email_notification(session_id, event_id, "exit", trigger_data)
                    
                    logger.info(f"✅ Event {event_id} recording started successfully")
                    return event_id
                else:
                    # Clean up failed event
                    del self.active_events[session_id]
                    logger.error(f"❌ Failed to start recording for event {event_id}")
                    return None
                    
        except Exception as e:
            logger.error(f"💥 Error starting event for session {session_id}: {e}")
            return None
    
    async def end_event(self, session_id: str, trigger_data: Dict[str, Any]) -> bool:
        """End the active event for the session"""
        try:
            with self.coordinator_lock:
                if session_id not in self.active_events:
                    logger.warning(f"⚠️ No active event found for session {session_id}")
                    return False
                
                event_data = self.active_events[session_id]
                event_id = event_data["event_id"]
                
                logger.info(f"🏁 Ending event {event_id} for session {session_id}")
                
                # Update event data
                event_data["end_time"] = datetime.now()
                event_data["end_trigger_data"] = trigger_data
                event_data["status"] = "ending"
                
                # Stop recording on both cameras
                recording_result = await self._stop_dual_camera_recording(session_id, event_id)
                
                if recording_result:
                    event_data["video_paths"].update(recording_result)
                    event_data["status"] = "completed"
                    
                    # Send event end notification to cameras
                    await self._notify_cameras_event_end(session_id, event_id, trigger_data)
                    
                    # Queue email notification
                    await self._queue_email_notification(session_id, event_id, "entrance", trigger_data)
                    
                    # Save event to database
                    await self._save_event_to_database(event_data)
                    
                    # Move to history and clean up
                    self.event_history.append(event_data.copy())
                    del self.active_events[session_id]
                    
                    logger.info(f"✅ Event {event_id} completed successfully")
                    return True
                else:
                    event_data["status"] = "error"
                    logger.error(f"❌ Failed to stop recording for event {event_id}")
                    return False
                    
        except Exception as e:
            logger.error(f"💥 Error ending event for session {session_id}: {e}")
            return False
    
    async def _start_dual_camera_recording(self, session_id: str, event_id: str) -> Optional[Dict[str, str]]:
        """Start recording on both internal and external cameras"""
        try:
            session = camera_session_manager.get_session(session_id)
            if not session:
                logger.error(f"Session {session_id} not found")
                return None
            
            # Set external camera configuration
            if session.external_camera_id:
                video_recording_service.set_external_camera_config(session.external_camera_id)
            
            # Start event recording
            recording_paths = video_recording_service.start_event_recording(event_id)
            
            if recording_paths:
                logger.info(f"📹 Recording started for event {event_id}")
                logger.info(f"   Internal: {recording_paths.get('internal_video', 'N/A')}")
                logger.info(f"   External: {recording_paths.get('external_video', 'N/A')}")
                
                return {
                    "internal_started": bool(recording_paths.get('internal_video')),
                    "external_started": bool(recording_paths.get('external_video')),
                    "start_time": datetime.now().isoformat()
                }
            else:
                logger.error(f"Failed to start recording for event {event_id}")
                return None
                
        except Exception as e:
            logger.error(f"💥 Error starting dual camera recording for event {event_id}: {e}")
            return None
    
    async def _stop_dual_camera_recording(self, session_id: str, event_id: str) -> Optional[Dict[str, str]]:
        """Stop recording on both cameras and get video paths"""
        try:
            # Stop event recording
            recording_result = video_recording_service.stop_event_recording()
            
            if recording_result:
                logger.info(f"📹 Recording stopped for event {event_id}")
                logger.info(f"   Internal video: {recording_result.get('internal_video', 'N/A')}")
                logger.info(f"   External video: {recording_result.get('external_video', 'N/A')}")
                
                return {
                    "internal_video_path": recording_result.get('internal_video'),
                    "external_video_path": recording_result.get('external_video'),
                    "stop_time": datetime.now().isoformat()
                }
            else:
                logger.error(f"Failed to stop recording for event {event_id}")
                return None
                
        except Exception as e:
            logger.error(f"💥 Error stopping dual camera recording for event {event_id}: {e}")
            return None
    
    async def _notify_cameras_event_start(self, session_id: str, event_id: str, trigger_data: Dict[str, Any]):
        """Notify both cameras about event start"""
        try:
            notification = {
                "type": "event_started",
                "event_id": event_id,
                "session_id": session_id,
                "timestamp": datetime.now().isoformat(),
                "bee_status": trigger_data.get("bee_status", "unknown"),
                "recording_active": True
            }
            
            await websocket_connection_manager.broadcast_to_session(session_id, notification)
            logger.info(f"📢 Event start notification sent to session {session_id}")
            
        except Exception as e:
            logger.error(f"💥 Error notifying cameras about event start: {e}")
    
    async def _notify_cameras_event_end(self, session_id: str, event_id: str, trigger_data: Dict[str, Any]):
        """Notify both cameras about event end"""
        try:
            notification = {
                "type": "event_ended",
                "event_id": event_id,
                "session_id": session_id,
                "timestamp": datetime.now().isoformat(),
                "bee_status": trigger_data.get("bee_status", "unknown"),
                "recording_active": False
            }
            
            await websocket_connection_manager.broadcast_to_session(session_id, notification)
            logger.info(f"📢 Event end notification sent to session {session_id}")
            
        except Exception as e:
            logger.error(f"💥 Error notifying cameras about event end: {e}")
    
    async def _queue_email_notification(self, session_id: str, event_id: str, event_type: str, trigger_data: Dict[str, Any]):
        """Queue email notification for the event"""
        try:
            notification_data = {
                "session_id": session_id,
                "event_id": event_id,
                "event_type": event_type,  # 'exit' or 'entrance'
                "timestamp": datetime.now(),
                "trigger_data": trigger_data,
                "retry_count": 0
            }
            
            await self.notification_queue.put(notification_data)
            logger.info(f"📧 Email notification queued for event {event_id} ({event_type})")
            
        except Exception as e:
            logger.error(f"💥 Error queuing email notification: {e}")
    
    async def _save_event_to_database(self, event_data: Dict[str, Any]):
        """Save completed event to database"""
        try:
            start_time = event_data.get("start_time", datetime.now())
            end_time = event_data.get("end_time", datetime.now())
            
            # Create database event record
            event_create = EventCreate(
                time_out=start_time,
                time_in=end_time,
                internal_video_url=event_data["video_paths"].get("internal_video_path"),
                external_video_url=event_data["video_paths"].get("external_video_path"),
                internal_video_url_converted=None,  # Will be set by conversion process
                external_video_url_converted=None,  # Will be set by conversion process
                conversion_status="pending",
                conversion_error=None
            )
            
            # Save to database
            db_event = await create_event(event_create)
            
            logger.info(f"💾 Event {event_data['event_id']} saved to database with ID: {db_event.id}")
            
        except Exception as e:
            logger.error(f"💥 Error saving event to database: {e}")
    
    async def process_email_notifications(self):
        """Process queued email notifications (background task)"""
        logger.info("📧 Starting email notification processor")
        
        while True:
            try:
                # Wait for notification with timeout
                notification_data = await asyncio.wait_for(
                    self.notification_queue.get(), 
                    timeout=30.0
                )
                
                # Send email notification
                success = await self._send_email_notification(notification_data)
                
                if not success and notification_data["retry_count"] < 3:
                    # Retry failed notifications
                    notification_data["retry_count"] += 1
                    await asyncio.sleep(60 * notification_data["retry_count"])  # Exponential backoff
                    await self.notification_queue.put(notification_data)
                    logger.info(f"📧 Retrying email notification (attempt {notification_data['retry_count'] + 1})")
                
                # Mark task as done
                self.notification_queue.task_done()
                
            except asyncio.TimeoutError:
                # No notifications to process, continue
                continue
            except Exception as e:
                logger.error(f"💥 Error processing email notifications: {e}")
                await asyncio.sleep(10)  # Wait before continuing
    
    async def _send_email_notification(self, notification_data: Dict[str, Any]) -> bool:
        """Send email notification for event"""
        try:
            event_type = notification_data["event_type"]
            timestamp = notification_data["timestamp"]
            
            # Send email using email service
            success = email_service.send_bee_detection_notification(
                event_type=event_type,
                timestamp=timestamp,
                additional_info={
                    "session_id": notification_data["session_id"],
                    "event_id": notification_data["event_id"],
                    "trigger_data": notification_data["trigger_data"]
                }
            )
            
            if success:
                logger.info(f"✅ Email notification sent for event {notification_data['event_id']} ({event_type})")
            else:
                logger.error(f"❌ Failed to send email notification for event {notification_data['event_id']}")
            
            return success
            
        except Exception as e:
            logger.error(f"💥 Error sending email notification: {e}")
            return False
    
    def get_active_events(self) -> Dict[str, Any]:
        """Get all active events"""
        with self.coordinator_lock:
            return {
                "active_events": dict(self.active_events),
                "total_active": len(self.active_events),
                "coordinator_status": "active" if self.active_events else "idle"
            }
    
    def get_event_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent event history"""
        return self.event_history[-limit:] if limit > 0 else self.event_history
    
    def get_session_event_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get event status for specific session"""
        with self.coordinator_lock:
            if session_id in self.active_events:
                event_data = self.active_events[session_id]
                return {
                    "has_active_event": True,
                    "event_id": event_data["event_id"],
                    "status": event_data["status"],
                    "start_time": event_data["start_time"],
                    "bee_status": event_data["bee_status"],
                    "cameras_recording": event_data["cameras_recording"]
                }
            else:
                return {
                    "has_active_event": False,
                    "last_event": self._get_last_event_for_session(session_id)
                }
    
    def _get_last_event_for_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get the last event for a session from history"""
        for event in reversed(self.event_history):
            if event.get("session_id") == session_id:
                return {
                    "event_id": event["event_id"],
                    "end_time": event.get("end_time"),
                    "status": event["status"]
                }
        return None
    
    async def cancel_event(self, session_id: str, reason: str = "manual_cancellation") -> bool:
        """Cancel an active event"""
        try:
            with self.coordinator_lock:
                if session_id not in self.active_events:
                    logger.warning(f"⚠️ No active event to cancel for session {session_id}")
                    return False
                
                event_data = self.active_events[session_id]
                event_id = event_data["event_id"]
                
                logger.info(f"🚫 Cancelling event {event_id} for session {session_id}: {reason}")
                
                # Stop recording
                await self._stop_dual_camera_recording(session_id, event_id)
                
                # Update event data
                event_data["status"] = "cancelled"
                event_data["cancellation_reason"] = reason
                event_data["end_time"] = datetime.now()
                
                # Notify cameras
                await websocket_connection_manager.broadcast_to_session(session_id, {
                    "type": "event_cancelled",
                    "event_id": event_id,
                    "reason": reason,
                    "timestamp": datetime.now().isoformat()
                })
                
                # Move to history and clean up
                self.event_history.append(event_data.copy())
                del self.active_events[session_id]
                
                logger.info(f"✅ Event {event_id} cancelled successfully")
                return True
                
        except Exception as e:
            logger.error(f"💥 Error cancelling event for session {session_id}: {e}")
            return False
    
    def get_coordinator_stats(self) -> Dict[str, Any]:
        """Get coordinator statistics"""
        with self.coordinator_lock:
            return {
                "active_events_count": len(self.active_events),
                "total_events_processed": len(self.event_history),
                "pending_notifications": self.notification_queue.qsize(),
                "coordinator_status": "healthy",
                "uptime": "active",  # Could track actual uptime if needed
                "last_event_time": self.event_history[-1].get("start_time") if self.event_history else None
            }

# Global singleton instance
event_coordinator = EventCoordinator() 