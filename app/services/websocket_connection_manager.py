"""
WebSocket Connection Manager - Manages dual WebSocket connections for camera sessions
"""
import asyncio
import json
from datetime import datetime
from typing import Dict, Optional, List, Any
from threading import RLock
from fastapi import WebSocket
from starlette.websockets import WebSocketDisconnect
import logging

from app.services.camera_session_manager import camera_session_manager, CameraSession

logger = logging.getLogger(__name__)

class WebSocketConnectionManager:
    """Manages WebSocket connections for camera sessions"""
    
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
            self.active_connections: Dict[str, Dict[str, WebSocket]] = {}
            self.connection_metadata: Dict[str, Dict[str, Dict]] = {}
            self.connection_lock = RLock()
            self._initialized = True
            logger.info("🔌 WebSocketConnectionManager initialized")
    
    async def register_connection(self, session_id: str, connection_type: str, websocket: WebSocket) -> bool:
        """Register a WebSocket connection for a session"""
        try:
            with self.connection_lock:
                # Initialize session connections if not exists
                if session_id not in self.active_connections:
                    self.active_connections[session_id] = {}
                    self.connection_metadata[session_id] = {}
                
                # Store the connection
                self.active_connections[session_id][connection_type] = websocket
                self.connection_metadata[session_id][connection_type] = {
                    "connected_at": datetime.now(),
                    "last_activity": datetime.now(),
                    "messages_sent": 0,
                    "messages_received": 0,
                    "status": "connected"
                }
                
                # Register with session manager
                success = camera_session_manager.register_connection(session_id, connection_type, websocket)
                
                if success:
                    logger.info(f"✅ {connection_type.title()} camera connected to session {session_id}")
                    return True
                else:
                    # Clean up if session registration failed
                    del self.active_connections[session_id][connection_type]
                    del self.connection_metadata[session_id][connection_type]
                    logger.error(f"❌ Failed to register {connection_type} connection for session {session_id}")
                    return False
                    
        except Exception as e:
            logger.error(f"💥 Error registering {connection_type} connection for session {session_id}: {e}")
            return False
    
    async def unregister_connection(self, session_id: str, connection_type: str) -> bool:
        """Unregister a WebSocket connection"""
        try:
            with self.connection_lock:
                # Remove from active connections
                if session_id in self.active_connections:
                    if connection_type in self.active_connections[session_id]:
                        del self.active_connections[session_id][connection_type]
                    
                    if connection_type in self.connection_metadata[session_id]:
                        del self.connection_metadata[session_id][connection_type]
                    
                    # Clean up empty session entries
                    if not self.active_connections[session_id]:
                        del self.active_connections[session_id]
                        del self.connection_metadata[session_id]
                
                # Unregister from session manager
                camera_session_manager.unregister_connection(session_id, connection_type)
                
                logger.info(f"🔌 {connection_type.title()} camera disconnected from session {session_id}")
                return True
                
        except Exception as e:
            logger.error(f"💥 Error unregistering {connection_type} connection for session {session_id}: {e}")
            return False
    
    def get_connection(self, session_id: str, connection_type: str) -> Optional[WebSocket]:
        """Get WebSocket connection for session and type"""
        with self.connection_lock:
            return self.active_connections.get(session_id, {}).get(connection_type)
    
    async def send_to_connection(self, session_id: str, connection_type: str, message: Any) -> bool:
        """Send message to specific connection"""
        try:
            websocket = self.get_connection(session_id, connection_type)
            if not websocket:
                logger.warning(f"⚠️ No {connection_type} connection found for session {session_id}")
                return False
            
            # Convert message to JSON if it's a dict
            if isinstance(message, dict):
                message_text = json.dumps(message, ensure_ascii=False)
            else:
                message_text = str(message)
            
            await websocket.send_text(message_text)
            
            # Update metadata
            with self.connection_lock:
                if session_id in self.connection_metadata and connection_type in self.connection_metadata[session_id]:
                    metadata = self.connection_metadata[session_id][connection_type]
                    metadata["last_activity"] = datetime.now()
                    metadata["messages_sent"] += 1
            
            logger.debug(f"📤 Message sent to {connection_type} camera in session {session_id}")
            return True
            
        except WebSocketDisconnect:
            logger.info(f"🔌 {connection_type.title()} camera disconnected from session {session_id}")
            await self.unregister_connection(session_id, connection_type)
            return False
        except Exception as e:
            logger.error(f"💥 Error sending message to {connection_type} camera in session {session_id}: {e}")
            return False
    
    async def broadcast_to_session(self, session_id: str, message: Any) -> Dict[str, bool]:
        """Broadcast message to all connections in a session"""
        results = {}
        
        with self.connection_lock:
            session_connections = self.active_connections.get(session_id, {})
        
        for connection_type in session_connections.keys():
            success = await self.send_to_connection(session_id, connection_type, message)
            results[connection_type] = success
        
        logger.debug(f"📡 Broadcast to session {session_id}: {results}")
        return results
    
    async def send_bytes_to_connection(self, session_id: str, connection_type: str, data: bytes) -> bool:
        """Send binary data to specific connection"""
        try:
            websocket = self.get_connection(session_id, connection_type)
            if not websocket:
                return False
            
            await websocket.send_bytes(data)
            
            # Update metadata
            with self.connection_lock:
                if session_id in self.connection_metadata and connection_type in self.connection_metadata[session_id]:
                    metadata = self.connection_metadata[session_id][connection_type]
                    metadata["last_activity"] = datetime.now()
                    metadata["messages_sent"] += 1
            
            return True
            
        except WebSocketDisconnect:
            await self.unregister_connection(session_id, connection_type)
            return False
        except Exception as e:
            logger.error(f"💥 Error sending bytes to {connection_type} camera in session {session_id}: {e}")
            return False
    
    def get_connection_status(self, session_id: str) -> Dict[str, Any]:
        """Get connection status for a session"""
        with self.connection_lock:
            session_connections = self.active_connections.get(session_id, {})
            session_metadata = self.connection_metadata.get(session_id, {})
            
            status = {
                "session_id": session_id,
                "connections": {},
                "total_connections": len(session_connections),
                "fully_connected": len(session_connections) >= 2
            }
            
            for conn_type, websocket in session_connections.items():
                metadata = session_metadata.get(conn_type, {})
                status["connections"][conn_type] = {
                    "connected": True,
                    "ready_state": websocket.client_state.name if hasattr(websocket, 'client_state') else "unknown",
                    "connected_at": metadata.get("connected_at", "unknown"),
                    "last_activity": metadata.get("last_activity", "unknown"),
                    "messages_sent": metadata.get("messages_sent", 0),
                    "messages_received": metadata.get("messages_received", 0)
                }
            
            return status
    
    def cleanup_orphaned_connections(self) -> int:
        """Clean up connections that no longer have valid sessions"""
        cleaned_count = 0
        
        with self.connection_lock:
            # Get list of session IDs that no longer exist in session manager
            active_session_ids = camera_session_manager.list_active_sessions()
            connection_session_ids = list(self.active_connections.keys())
            
            orphaned_sessions = [sid for sid in connection_session_ids if sid not in active_session_ids]
            
            for session_id in orphaned_sessions:
                logger.info(f"🧹 Cleaning up orphaned connections for session {session_id}")
                
                # Close all connections in the orphaned session
                session_connections = self.active_connections.get(session_id, {})
                for connection_type, websocket in session_connections.items():
                    try:
                        asyncio.create_task(websocket.close())
                    except Exception as e:
                        logger.error(f"Error closing orphaned {connection_type} connection: {e}")
                
                # Remove from tracking
                if session_id in self.active_connections:
                    del self.active_connections[session_id]
                if session_id in self.connection_metadata:
                    del self.connection_metadata[session_id]
                
                cleaned_count += len(session_connections)
        
        if cleaned_count > 0:
            logger.info(f"🧹 Cleaned up {cleaned_count} orphaned connections")
        
        return cleaned_count
    
    def get_all_connections_status(self) -> Dict[str, Any]:
        """Get status of all connections"""
        with self.connection_lock:
            total_sessions = len(self.active_connections)
            total_connections = sum(len(connections) for connections in self.active_connections.values())
            
            sessions_status = {}
            for session_id in self.active_connections:
                sessions_status[session_id] = self.get_connection_status(session_id)
            
            return {
                "total_sessions": total_sessions,
                "total_connections": total_connections,
                "sessions": sessions_status,
                "manager_status": "healthy" if total_sessions > 0 else "idle"
            }
    
    async def perform_session_handshake(self, session_id: str, connection_type: str, websocket: WebSocket) -> bool:
        """Perform handshake protocol with client"""
        try:
            # Send handshake request
            handshake_request = {
                "type": "session_handshake_request",
                "session_id": session_id,
                "connection_type": connection_type,
                "server_timestamp": datetime.now().isoformat(),
                "expected_response": "session_handshake_response"
            }
            
            await websocket.send_text(json.dumps(handshake_request))
            logger.debug(f"🤝 Handshake request sent to {connection_type} camera in session {session_id}")
            
            # Wait for handshake response (with timeout)
            try:
                response = await asyncio.wait_for(websocket.receive_text(), timeout=10.0)
                response_data = json.loads(response)
                
                if (response_data.get("type") == "session_handshake_response" and
                    response_data.get("session_id") == session_id and
                    response_data.get("connection_type") == connection_type):
                    
                    # Mark connection as ready
                    camera_session_manager.mark_connection_ready(session_id, connection_type)
                    
                    # Send handshake confirmation
                    confirmation = {
                        "type": "session_handshake_confirmed",
                        "session_id": session_id,
                        "connection_type": connection_type,
                        "status": "ready",
                        "server_timestamp": datetime.now().isoformat()
                    }
                    await websocket.send_text(json.dumps(confirmation))
                    
                    logger.info(f"✅ Handshake completed for {connection_type} camera in session {session_id}")
                    return True
                else:
                    logger.error(f"❌ Invalid handshake response from {connection_type} camera in session {session_id}")
                    return False
                    
            except asyncio.TimeoutError:
                logger.error(f"⏰ Handshake timeout for {connection_type} camera in session {session_id}")
                return False
            except json.JSONDecodeError:
                logger.error(f"❌ Invalid JSON in handshake response from {connection_type} camera in session {session_id}")
                return False
                
        except Exception as e:
            logger.error(f"💥 Handshake error for {connection_type} camera in session {session_id}: {e}")
            return False
    
    async def close_session_connections(self, session_id: str):
        """Close all connections for a session"""
        with self.connection_lock:
            session_connections = self.active_connections.get(session_id, {}).copy()
        
        for connection_type, websocket in session_connections.items():
            try:
                await websocket.close()
                logger.info(f"🔌 Closed {connection_type} connection for session {session_id}")
            except Exception as e:
                logger.error(f"Error closing {connection_type} connection for session {session_id}: {e}")
            
            await self.unregister_connection(session_id, connection_type)

# Global singleton instance
websocket_connection_manager = WebSocketConnectionManager() 