"""
Camera Session Manager - Enhanced with error handling and recovery mechanisms
"""
import asyncio
import logging
import time
from typing import Dict, Optional, Set, List
from dataclasses import dataclass, field
from datetime import datetime
import uuid
from enum import Enum

from .error_handler import error_handler, ErrorType, ErrorSeverity

logger = logging.getLogger(__name__)

class SessionStatus(Enum):
    INITIALIZING = "initializing"
    READY = "ready"
    ACTIVE = "active"
    ERROR = "error"
    TERMINATED = "terminated"

@dataclass
class CameraSession:
    """Enhanced camera session with error handling"""
    session_id: str
    internal_camera_id: str
    external_camera_id: str
    created_at: datetime
    status: SessionStatus = SessionStatus.INITIALIZING
    
    # Connection status
    internal_connected: bool = False
    external_connected: bool = False
    
    # Error handling
    error_count: int = 0
    last_error: Optional[str] = None
    recovery_attempts: int = 0
    max_recovery_attempts: int = 3
    
    # Activity tracking
    last_activity: datetime = field(default_factory=datetime.now)
    frame_count: int = 0
    
    # Session metadata
    metadata: Dict[str, any] = field(default_factory=dict)
    
    @property
    def is_fully_connected(self) -> bool:
        return self.internal_connected and self.external_connected
    
    @property
    def is_ready_for_recording(self) -> bool:
        return self.status == SessionStatus.READY and self.is_fully_connected
    
    @property
    def connection_health(self) -> float:
        """Calculate connection health score (0-1)"""
        health_score = 0.0
        
        # Base connection score
        if self.internal_connected:
            health_score += 0.5
        if self.external_connected:
            health_score += 0.5
            
        # Penalty for errors
        if self.error_count > 0:
            health_score *= max(0.1, 1 - (self.error_count * 0.1))
            
        # Penalty for recovery attempts
        if self.recovery_attempts > 0:
            health_score *= max(0.2, 1 - (self.recovery_attempts * 0.2))
        
        return health_score

class CameraSessionManager:
    """Enhanced Session Manager with error handling and recovery"""
    
    def __init__(self):
        self.sessions: Dict[str, CameraSession] = {}
        self.session_timeout = 3600  # 1 hour
        self.cleanup_interval = 300  # 5 minutes
        self.max_sessions = 10
        
        # Error tracking
        self.global_error_count = 0
        self.recovery_callbacks = {}
        
        # Background tasks
        self.cleanup_task = None
        self.health_check_task = None
        
        # Register error recovery callbacks
        self._register_recovery_callbacks()
        
        logger.info("📹 Enhanced Camera Session Manager initialized")
        
        # Start background tasks
        self._start_background_tasks()
    
    def _register_recovery_callbacks(self):
        """Register error recovery callbacks"""
        
        async def camera_disconnection_recovery(error_info):
            """Recovery for camera disconnection errors"""
            session_id = error_info.session_id
            camera_type = error_info.camera_type
            
            if not session_id or session_id not in self.sessions:
                return False
                
            session = self.sessions[session_id]
            
            logger.info(f"🔄 Attempting camera recovery for session {session_id}, camera {camera_type}")
            
            # Reset camera connection status
            if camera_type == "internal":
                session.internal_connected = False
            elif camera_type == "external":
                session.external_connected = False
            
            # Increment recovery attempts
            session.recovery_attempts += 1
            
            # If too many recovery attempts, mark session as error
            if session.recovery_attempts >= session.max_recovery_attempts:
                logger.error(f"Max recovery attempts reached for session {session_id}")
                session.status = SessionStatus.ERROR
                return False
            
            # Wait for new connection attempt
            await asyncio.sleep(2)
            
            # Check if camera reconnected
            if camera_type == "internal" and session.internal_connected:
                logger.info(f"✅ Internal camera recovered for session {session_id}")
                return True
            elif camera_type == "external" and session.external_connected:
                logger.info(f"✅ External camera recovered for session {session_id}")
                return True
            
            return False
        
        async def session_error_recovery(error_info):
            """Recovery for session errors"""
            session_id = error_info.session_id
            
            if not session_id or session_id not in self.sessions:
                return False
                
            session = self.sessions[session_id]
            
            logger.info(f"🔄 Attempting session recovery for {session_id}")
            
            # Reset session to initializing state
            session.status = SessionStatus.INITIALIZING
            session.error_count += 1
            session.last_error = error_info.message
            
            # Attempt to reinitialize session
            try:
                await self._reinitialize_session(session)
                return True
            except Exception as e:
                logger.error(f"Session recovery failed: {e}")
                return False
        
        # Register callbacks
        error_handler.register_recovery_callback(ErrorType.CAMERA_DISCONNECTION, camera_disconnection_recovery)
        error_handler.register_recovery_callback(ErrorType.SESSION_ERROR, session_error_recovery)
    
    async def _reinitialize_session(self, session: CameraSession):
        """Reinitialize a session after error"""
        logger.info(f"Reinitializing session {session.session_id}")
        
        # Reset connection status
        session.internal_connected = False
        session.external_connected = False
        session.status = SessionStatus.INITIALIZING
        
        # Update activity
        session.last_activity = datetime.now()
        
        # Session will be marked as ready when cameras reconnect
        logger.info(f"Session {session.session_id} reinitialized - waiting for camera connections")
    
    def _start_background_tasks(self):
        """Start background maintenance tasks"""
        self.cleanup_task = asyncio.create_task(self._cleanup_expired_sessions())
        self.health_check_task = asyncio.create_task(self._health_check_loop())
    
    async def _cleanup_expired_sessions(self):
        """Clean up expired sessions"""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval)
                
                current_time = datetime.now()
                expired_sessions = []
                
                for session_id, session in self.sessions.items():
                    time_since_activity = (current_time - session.last_activity).total_seconds()
                    
                    if time_since_activity > self.session_timeout:
                        expired_sessions.append(session_id)
                        logger.info(f"Session {session_id} expired (inactive for {time_since_activity:.1f}s)")
                
                # Remove expired sessions
                for session_id in expired_sessions:
                    await self.cleanup_session(session_id)
                
            except Exception as e:
                logger.error(f"Error in cleanup task: {e}")
                await asyncio.sleep(60)  # Wait before retrying
    
    async def _health_check_loop(self):
        """Continuous health monitoring"""
        while True:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                for session_id, session in self.sessions.items():
                    await self._check_session_health(session)
                
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
                await asyncio.sleep(60)
    
    async def _check_session_health(self, session: CameraSession):
        """Check individual session health"""
        try:
            health_score = session.connection_health
            
            if health_score < 0.5:
                logger.warning(f"Session {session.session_id} has low health score: {health_score:.2f}")
                
                # Report session health issue
                await error_handler.handle_error(
                    ErrorType.SESSION_ERROR,
                    f"Session health degraded: {health_score:.2f}",
                    ErrorSeverity.MEDIUM,
                    session_id=session.session_id,
                    metadata={"health_score": health_score}
                )
            
            # Check for stale sessions (no activity for too long)
            time_since_activity = (datetime.now() - session.last_activity).total_seconds()
            if time_since_activity > 300:  # 5 minutes
                logger.warning(f"Session {session.session_id} has been inactive for {time_since_activity:.1f}s")
                
                await error_handler.handle_error(
                    ErrorType.SESSION_ERROR,
                    f"Session inactive for {time_since_activity:.1f}s",
                    ErrorSeverity.LOW,
                    session_id=session.session_id,
                    metadata={"inactive_seconds": time_since_activity}
                )
            
        except Exception as e:
            logger.error(f"Error checking session health: {e}")
    
    async def create_session(self, internal_camera_id: str, external_camera_id: str) -> CameraSession:
        """Create a new camera session with error handling"""
        try:
            # Check session limit
            if len(self.sessions) >= self.max_sessions:
                await error_handler.handle_error(
                    ErrorType.SESSION_ERROR,
                    f"Maximum sessions ({self.max_sessions}) exceeded",
                    ErrorSeverity.HIGH,
                    metadata={"current_sessions": len(self.sessions)}
                )
                raise ValueError("Maximum sessions exceeded")
            
            # Generate session ID
            session_id = f"session_{uuid.uuid4().hex[:8]}"
            
            # Create session
            session = CameraSession(
                session_id=session_id,
                internal_camera_id=internal_camera_id,
                external_camera_id=external_camera_id,
                created_at=datetime.now(),
                status=SessionStatus.INITIALIZING
            )
            
            # Store session
            self.sessions[session_id] = session
            
            logger.info(f"✅ Session {session_id} created successfully")
            return session
            
        except Exception as e:
            await error_handler.handle_error(
                ErrorType.SESSION_ERROR,
                f"Failed to create session: {str(e)}",
                ErrorSeverity.HIGH,
                exception=e,
                metadata={
                    "internal_camera_id": internal_camera_id,
                    "external_camera_id": external_camera_id
                }
            )
            raise
    
    async def register_connection(self, session_id: str, camera_type: str, connection_id: str) -> bool:
        """Register a camera connection with error handling"""
        try:
            if session_id not in self.sessions:
                await error_handler.handle_error(
                    ErrorType.SESSION_ERROR,
                    f"Session {session_id} not found for connection registration",
                    ErrorSeverity.MEDIUM,
                    session_id=session_id,
                    camera_type=camera_type
                )
                return False
            
            session = self.sessions[session_id]
            
            # Update connection status
            if camera_type == "internal":
                session.internal_connected = True
            elif camera_type == "external":
                session.external_connected = True
            else:
                await error_handler.handle_error(
                    ErrorType.SESSION_ERROR,
                    f"Invalid camera type: {camera_type}",
                    ErrorSeverity.MEDIUM,
                    session_id=session_id,
                    camera_type=camera_type
                )
                return False
            
            # Update session activity
            session.last_activity = datetime.now()
            
            # Update session status
            if session.is_fully_connected:
                session.status = SessionStatus.READY
                logger.info(f"✅ Session {session_id} is fully connected and ready")
            
            # Reset error count on successful connection
            session.error_count = 0
            session.recovery_attempts = 0
            
            logger.info(f"📹 {camera_type.title()} camera connected to session {session_id}")
            return True
            
        except Exception as e:
            await error_handler.handle_error(
                ErrorType.CAMERA_DISCONNECTION,
                f"Failed to register {camera_type} camera connection: {str(e)}",
                ErrorSeverity.HIGH,
                session_id=session_id,
                camera_type=camera_type,
                exception=e
            )
            return False
    
    async def unregister_connection(self, session_id: str, camera_type: str) -> bool:
        """Unregister a camera connection with error handling"""
        try:
            if session_id not in self.sessions:
                logger.warning(f"Session {session_id} not found for connection unregistration")
                return False
            
            session = self.sessions[session_id]
            
            # Update connection status
            if camera_type == "internal":
                session.internal_connected = False
            elif camera_type == "external":
                session.external_connected = False
            
            # Update session activity
            session.last_activity = datetime.now()
            
            # Check if session is still viable
            if not session.is_fully_connected:
                session.status = SessionStatus.INITIALIZING
                
                # Report camera disconnection
                await error_handler.handle_error(
                    ErrorType.CAMERA_DISCONNECTION,
                    f"{camera_type.title()} camera disconnected",
                    ErrorSeverity.MEDIUM,
                    session_id=session_id,
                    camera_type=camera_type
                )
            
            logger.info(f"🔌 {camera_type.title()} camera disconnected from session {session_id}")
            return True
            
        except Exception as e:
            await error_handler.handle_error(
                ErrorType.CAMERA_DISCONNECTION,
                f"Failed to unregister {camera_type} camera connection: {str(e)}",
                ErrorSeverity.HIGH,
                session_id=session_id,
                camera_type=camera_type,
                exception=e
            )
            return False
    
    async def update_session_activity(self, session_id: str):
        """Update session activity timestamp"""
        if session_id in self.sessions:
            session = self.sessions[session_id]
            session.last_activity = datetime.now()
            session.frame_count += 1
            
            # Update session status if needed
            if session.status == SessionStatus.INITIALIZING and session.is_fully_connected:
                session.status = SessionStatus.READY
    
    async def get_session_status(self, session_id: str) -> Optional[Dict]:
        """Get detailed session status"""
        if session_id not in self.sessions:
            return None
        
        session = self.sessions[session_id]
        
        return {
            "session_id": session.session_id,
            "status": session.status.value,
            "internal_connected": session.internal_connected,
            "external_connected": session.external_connected,
            "is_fully_connected": session.is_fully_connected,
            "is_ready_for_recording": session.is_ready_for_recording,
            "connection_health": session.connection_health,
            "created_at": session.created_at.isoformat(),
            "last_activity": session.last_activity.isoformat(),
            "frame_count": session.frame_count,
            "error_count": session.error_count,
            "recovery_attempts": session.recovery_attempts,
            "last_error": session.last_error
        }
    
    async def cleanup_session(self, session_id: str) -> bool:
        """Clean up a session with error handling"""
        try:
            if session_id not in self.sessions:
                logger.warning(f"Session {session_id} not found for cleanup")
                return False
            
            session = self.sessions[session_id]
            session.status = SessionStatus.TERMINATED
            
            # Remove from sessions
            del self.sessions[session_id]
            
            logger.info(f"🧹 Session {session_id} cleaned up successfully")
            return True
            
        except Exception as e:
            await error_handler.handle_error(
                ErrorType.SESSION_ERROR,
                f"Failed to cleanup session {session_id}: {str(e)}",
                ErrorSeverity.MEDIUM,
                session_id=session_id,
                exception=e
            )
            return False
    
    def get_system_health(self) -> Dict:
        """Get overall system health"""
        active_sessions = len(self.sessions)
        healthy_sessions = sum(1 for s in self.sessions.values() if s.connection_health > 0.7)
        
        return {
            "active_sessions": active_sessions,
            "healthy_sessions": healthy_sessions,
            "max_sessions": self.max_sessions,
            "system_health": healthy_sessions / max(1, active_sessions),
            "global_error_count": self.global_error_count,
            "error_metrics": error_handler.get_error_metrics()
        }
    
    async def shutdown(self):
        """Shutdown the session manager"""
        logger.info("🛑 Shutting down Camera Session Manager")
        
        # Cancel background tasks
        if self.cleanup_task:
            self.cleanup_task.cancel()
        if self.health_check_task:
            self.health_check_task.cancel()
        
        # Clean up all sessions
        for session_id in list(self.sessions.keys()):
            await self.cleanup_session(session_id)
        
        logger.info("✅ Camera Session Manager shutdown complete")

# Create singleton instance
camera_session_manager = CameraSessionManager() 