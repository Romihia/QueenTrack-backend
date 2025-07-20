"""
Services package initialization
"""
from .bee_tracking_service import bee_tracking_service
from .video_recording_service import video_recording_service
from .video_service import video_service
from .email_service import email_service

__all__ = [
    'bee_tracking_service',
    'video_recording_service', 
    'video_service',
    'email_service'
]

"""
Service Layer Initialization - Singleton instances for session-based architecture
"""

# Import all services
from .camera_session_manager import CameraSessionManager
from .websocket_connection_manager import WebSocketConnectionManager
from .dual_camera_websocket_handler import DualCameraWebSocketHandler
from .event_coordinator import EventCoordinator
from .camera_synchronizer import CameraSynchronizer

# Create singleton instances
camera_session_manager = CameraSessionManager()
websocket_connection_manager = WebSocketConnectionManager()
dual_camera_websocket_handler = DualCameraWebSocketHandler()
event_coordinator = EventCoordinator()

# Export instances
__all__ = [
    'camera_session_manager',
    'websocket_connection_manager', 
    'dual_camera_websocket_handler',
    'event_coordinator'
]
