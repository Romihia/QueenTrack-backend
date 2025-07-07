"""
Comprehensive tests for session management system
"""
import pytest
import asyncio
import unittest
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta
import json
import tempfile
import os
import sys

# Add the app directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.services.camera_session_manager import CameraSessionManager, CameraSession, SessionStatus
from app.services.error_handler import ErrorHandler, ErrorType, ErrorSeverity
from app.services.performance_monitor import PerformanceMonitor
from app.services.websocket_connection_manager import WebSocketConnectionManager
from app.services.event_coordinator import EventCoordinator

class TestCameraSessionManager:
    """Test cases for camera session management"""
    
    @pytest.fixture
    def session_manager(self):
        """Create a fresh session manager for each test"""
        return CameraSessionManager()
    
    @pytest.fixture
    def mock_websocket(self):
        """Mock WebSocket connection"""
        mock_ws = Mock()
        mock_ws.close = AsyncMock()
        return mock_ws
    
    @pytest.mark.asyncio
    async def test_create_session_success(self, session_manager):
        """Test successful session creation"""
        internal_camera = "test_internal_001"
        external_camera = "test_external_001"
        
        session = await session_manager.create_session(internal_camera, external_camera)
        
        assert session is not None
        assert session.internal_camera_id == internal_camera
        assert session.external_camera_id == external_camera
        assert session.status == SessionStatus.INITIALIZING
        assert not session.is_fully_connected
        assert session.session_id in session_manager.sessions
    
    @pytest.mark.asyncio
    async def test_session_connection_lifecycle(self, session_manager):
        """Test complete session connection lifecycle"""
        session = await session_manager.create_session("cam1", "cam2")
        session_id = session.session_id
        
        # Initially not connected
        assert not session.internal_connected
        assert not session.external_connected
        assert not session.is_fully_connected
        
        # Connect internal camera
        success = await session_manager.register_connection(session_id, "internal", "conn1")
        assert success
        assert session.internal_connected
        assert not session.is_fully_connected  # Still waiting for external
        
        # Connect external camera
        success = await session_manager.register_connection(session_id, "external", "conn2")
        assert success
        assert session.external_connected
        assert session.is_fully_connected
        assert session.status == SessionStatus.READY
    
    @pytest.mark.asyncio
    async def test_session_disconnection_handling(self, session_manager):
        """Test handling of camera disconnections"""
        session = await session_manager.create_session("cam1", "cam2")
        session_id = session.session_id
        
        # Connect both cameras
        await session_manager.register_connection(session_id, "internal", "conn1")
        await session_manager.register_connection(session_id, "external", "conn2")
        assert session.is_fully_connected
        
        # Disconnect internal camera
        success = await session_manager.unregister_connection(session_id, "internal")
        assert success
        assert not session.internal_connected
        assert not session.is_fully_connected
        assert session.status == SessionStatus.INITIALIZING
    
    @pytest.mark.asyncio
    async def test_session_activity_tracking(self, session_manager):
        """Test session activity tracking"""
        session = await session_manager.create_session("cam1", "cam2")
        session_id = session.session_id
        
        initial_activity = session.last_activity
        initial_frames = session.frame_count
        
        # Simulate activity
        await asyncio.sleep(0.1)  # Small delay
        await session_manager.update_session_activity(session_id)
        
        assert session.last_activity > initial_activity
        assert session.frame_count == initial_frames + 1
    
    @pytest.mark.asyncio
    async def test_session_cleanup(self, session_manager):
        """Test session cleanup"""
        session = await session_manager.create_session("cam1", "cam2")
        session_id = session.session_id
        
        # Verify session exists
        assert session_id in session_manager.sessions
        
        # Cleanup session
        success = await session_manager.cleanup_session(session_id)
        assert success
        assert session_id not in session_manager.sessions
    
    @pytest.mark.asyncio
    async def test_max_sessions_limit(self, session_manager):
        """Test maximum sessions limit enforcement"""
        # Set low limit for testing
        session_manager.max_sessions = 2
        
        # Create maximum number of sessions
        session1 = await session_manager.create_session("cam1", "cam2")
        session2 = await session_manager.create_session("cam3", "cam4")
        
        assert len(session_manager.sessions) == 2
        
        # Try to exceed limit
        with pytest.raises(ValueError, match="Maximum sessions exceeded"):
            await session_manager.create_session("cam5", "cam6")
    
    @pytest.mark.asyncio
    async def test_session_health_monitoring(self, session_manager):
        """Test session health monitoring"""
        session = await session_manager.create_session("cam1", "cam2")
        
        # Initial health should be low (not connected)
        assert session.connection_health < 0.5
        
        # Connect cameras to improve health
        await session_manager.register_connection(session.session_id, "internal", "conn1")
        await session_manager.register_connection(session.session_id, "external", "conn2")
        
        # Health should improve
        assert session.connection_health >= 0.5
        
        # Simulate errors to degrade health
        session.error_count = 3
        assert session.connection_health < 0.5

class TestErrorHandler:
    """Test cases for error handling and recovery"""
    
    @pytest.fixture
    def error_handler(self):
        """Create a fresh error handler for each test"""
        return ErrorHandler()
    
    @pytest.mark.asyncio
    async def test_error_creation_and_tracking(self, error_handler):
        """Test error creation and tracking"""
        error_id = await error_handler.handle_error(
            ErrorType.CAMERA_DISCONNECTION,
            "Test camera disconnection",
            ErrorSeverity.MEDIUM,
            session_id="test_session",
            camera_type="internal"
        )
        
        assert error_id is not None
        assert error_id in error_handler.active_errors
        
        error_info = error_handler.get_error_status(error_id)
        assert error_info.error_type == ErrorType.CAMERA_DISCONNECTION
        assert error_info.message == "Test camera disconnection"
        assert error_info.session_id == "test_session"
        assert error_info.camera_type == "internal"
    
    @pytest.mark.asyncio
    async def test_error_recovery_callback(self, error_handler):
        """Test error recovery callback registration and execution"""
        recovery_called = False
        
        async def test_recovery_callback(error_info):
            nonlocal recovery_called
            recovery_called = True
            return True  # Indicate successful recovery
        
        # Register callback
        error_handler.register_recovery_callback(
            ErrorType.NETWORK_FAILURE, 
            test_recovery_callback
        )
        
        # Trigger error
        error_id = await error_handler.handle_error(
            ErrorType.NETWORK_FAILURE,
            "Test network failure",
            ErrorSeverity.HIGH
        )
        
        # Wait a bit for recovery to be attempted
        await asyncio.sleep(0.1)
        
        # Verify callback was called
        assert recovery_called
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_functionality(self, error_handler):
        """Test circuit breaker prevents repeated failures"""
        # Simulate multiple failures quickly
        for i in range(5):
            await error_handler.handle_error(
                ErrorType.CAMERA_DISCONNECTION,
                f"Repeated failure {i}",
                ErrorSeverity.HIGH,
                session_id="test_session"
            )
        
        # Circuit breaker should be open now
        assert len(error_handler.circuit_breakers) > 0
    
    def test_error_metrics_tracking(self, error_handler):
        """Test error metrics are properly tracked"""
        initial_metrics = error_handler.get_error_metrics()
        
        # The error handler should track various metrics
        assert "total_errors" in initial_metrics
        assert "errors_by_type" in initial_metrics
        assert "errors_by_severity" in initial_metrics
        assert "successful_recoveries" in initial_metrics
        assert "failed_recoveries" in initial_metrics

class TestPerformanceMonitor:
    """Test cases for performance monitoring"""
    
    @pytest.fixture
    def performance_monitor(self):
        """Create a fresh performance monitor for each test"""
        return PerformanceMonitor()
    
    @pytest.mark.asyncio
    async def test_performance_monitoring_lifecycle(self, performance_monitor):
        """Test starting and stopping performance monitoring"""
        assert not performance_monitor.monitoring_active
        
        # Start monitoring
        await performance_monitor.start_monitoring()
        assert performance_monitor.monitoring_active
        
        # Stop monitoring
        await performance_monitor.stop_monitoring()
        assert not performance_monitor.monitoring_active
    
    def test_session_metrics_recording(self, performance_monitor):
        """Test recording session-specific metrics"""
        session_id = "test_session_001"
        
        # Record some metrics
        performance_monitor.record_session_metric(
            session_id,
            frame_rate=25.0,
            processing_latency_ms=15.5,
            memory_usage_mb=128.0
        )
        
        # Verify metrics were recorded
        session_metrics = performance_monitor.get_session_metrics(session_id)
        assert session_metrics is not None
        assert session_metrics.frame_rate == 25.0
        assert session_metrics.processing_latency_ms == 15.5
        assert session_metrics.memory_usage_mb == 128.0
    
    def test_camera_metrics_recording(self, performance_monitor):
        """Test recording camera-specific metrics"""
        camera_id = "cam_001"
        
        # Record camera metrics
        performance_monitor.record_camera_metric(
            camera_id,
            "internal",
            fps=30.0,
            frame_drops=2,
            processing_time_ms=8.5
        )
        
        # Verify metrics were recorded
        camera_metrics = performance_monitor.get_camera_metrics(camera_id)
        assert camera_metrics is not None
        assert camera_metrics.fps == 30.0
        assert camera_metrics.frame_drops == 2
        assert camera_metrics.processing_time_ms == 8.5
    
    def test_performance_thresholds(self, performance_monitor):
        """Test performance threshold configuration"""
        # Set custom thresholds
        performance_monitor.set_thresholds(
            cpu_warning=70.0,
            memory_critical=85.0
        )
        
        assert performance_monitor.thresholds["cpu_warning"] == 70.0
        assert performance_monitor.thresholds["memory_critical"] == 85.0
    
    def test_alert_callback_registration(self, performance_monitor):
        """Test alert callback registration"""
        alert_received = False
        
        def test_alert_callback(alert_data):
            nonlocal alert_received
            alert_received = True
        
        # Register callback
        performance_monitor.add_alert_callback(test_alert_callback)
        
        # Verify callback was registered
        assert len(performance_monitor.alert_callbacks) > 0

class TestIntegration:
    """Integration tests for multiple components working together"""
    
    @pytest.mark.asyncio
    async def test_session_with_error_handling_integration(self):
        """Test session management with error handling integration"""
        session_manager = CameraSessionManager()
        
        # Create session
        session = await session_manager.create_session("cam1", "cam2")
        
        # Simulate connection and disconnection to trigger error handling
        await session_manager.register_connection(session.session_id, "internal", "conn1")
        await session_manager.unregister_connection(session.session_id, "internal")
        
        # Verify error handling was triggered (session should be in initializing state)
        assert session.status == SessionStatus.INITIALIZING
    
    @pytest.mark.asyncio
    async def test_performance_monitoring_with_sessions(self):
        """Test performance monitoring integration with session management"""
        session_manager = CameraSessionManager()
        performance_monitor = PerformanceMonitor()
        
        # Start monitoring
        await performance_monitor.start_monitoring()
        
        # Create and manage sessions while monitoring
        session = await session_manager.create_session("cam1", "cam2")
        
        # Record some metrics
        performance_monitor.record_session_metric(
            session.session_id,
            frame_rate=30.0,
            processing_latency_ms=12.0
        )
        
        # Verify metrics were recorded
        metrics = performance_monitor.get_session_metrics(session.session_id)
        assert metrics is not None
        assert metrics.frame_rate == 30.0
        
        # Stop monitoring
        await performance_monitor.stop_monitoring()

class TestConfigurationManagement:
    """Test configuration management functionality"""
    
    def test_configuration_export_import(self):
        """Test configuration export and import functionality"""
        # Create test configuration
        test_config = {
            "processing": {
                "detection_enabled": True,
                "detection_confidence_threshold": 0.25,
                "min_consecutive_detections": 3
            },
            "camera_config": {
                "internal_camera_id": "0",
                "external_camera_id": "1"
            }
        }
        
        # Test JSON serialization (export simulation)
        exported_json = json.dumps(test_config, indent=2)
        assert exported_json is not None
        
        # Test JSON deserialization (import simulation)
        imported_config = json.loads(exported_json)
        assert imported_config == test_config
    
    def test_configuration_validation(self):
        """Test configuration validation"""
        valid_config = {
            "processing": {
                "detection_confidence_threshold": 0.5,
                "min_consecutive_detections": 3
            }
        }
        
        # Valid configuration should have proper types and ranges
        assert isinstance(valid_config["processing"]["detection_confidence_threshold"], float)
        assert 0 <= valid_config["processing"]["detection_confidence_threshold"] <= 1
        assert isinstance(valid_config["processing"]["min_consecutive_detections"], int)
        assert valid_config["processing"]["min_consecutive_detections"] > 0

class TestVideoProcessing:
    """Test video processing and recording functionality"""
    
    @pytest.fixture
    def temp_video_dir(self):
        """Create temporary directory for video testing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    def test_video_path_generation(self, temp_video_dir):
        """Test video file path generation"""
        session_id = "test_session_001"
        timestamp = int(datetime.now().timestamp())
        
        # Generate expected paths
        internal_path = os.path.join(temp_video_dir, f"internal_camera_{timestamp}.mp4")
        external_path = os.path.join(temp_video_dir, f"external_camera_{timestamp}.mp4")
        
        # Verify paths are properly formatted
        assert internal_path.endswith(".mp4")
        assert external_path.endswith(".mp4")
        assert "internal_camera" in internal_path
        assert "external_camera" in external_path
    
    def test_video_metadata_structure(self):
        """Test video metadata structure"""
        metadata = {
            "session_id": "test_session",
            "camera_type": "internal",
            "start_time": datetime.now().isoformat(),
            "duration_seconds": 120.5,
            "frame_count": 3615,
            "resolution": "1920x1080",
            "fps": 30.0
        }
        
        # Verify required fields
        required_fields = ["session_id", "camera_type", "start_time"]
        for field in required_fields:
            assert field in metadata
        
        # Verify data types
        assert isinstance(metadata["duration_seconds"], (int, float))
        assert isinstance(metadata["frame_count"], int)
        assert isinstance(metadata["fps"], (int, float))

# Utility functions for testing
def create_mock_websocket():
    """Create a mock WebSocket for testing"""
    mock_ws = Mock()
    mock_ws.close = AsyncMock()
    mock_ws.send_text = AsyncMock()
    mock_ws.send_bytes = AsyncMock()
    return mock_ws

def create_mock_session_data():
    """Create mock session data for testing"""
    return {
        "session_id": "test_session_001",
        "internal_camera_id": "test_internal",
        "external_camera_id": "test_external",
        "created_at": datetime.now().isoformat()
    }

def assert_session_state(session, expected_state):
    """Assert session is in expected state"""
    state_checks = {
        "initializing": lambda s: s.status == SessionStatus.INITIALIZING,
        "ready": lambda s: s.status == SessionStatus.READY and s.is_fully_connected,
        "connected": lambda s: s.internal_connected and s.external_connected,
        "healthy": lambda s: s.connection_health > 0.7
    }
    
    if expected_state in state_checks:
        assert state_checks[expected_state](session), f"Session not in expected state: {expected_state}"

# Performance testing utilities
def measure_execution_time(func):
    """Decorator to measure function execution time"""
    import time
    import functools
    
    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        execution_time = time.time() - start_time
        print(f"{func.__name__} executed in {execution_time:.4f} seconds")
        return result
    
    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        print(f"{func.__name__} executed in {execution_time:.4f} seconds")
        return result
    
    return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

# Test data generators
def generate_test_sessions(count=5):
    """Generate test session data"""
    sessions = []
    for i in range(count):
        sessions.append({
            "session_id": f"test_session_{i:03d}",
            "internal_camera_id": f"internal_{i}",
            "external_camera_id": f"external_{i}",
            "created_at": datetime.now().isoformat()
        })
    return sessions

def generate_performance_data(duration_minutes=1):
    """Generate mock performance data"""
    data_points = []
    start_time = datetime.now()
    
    for i in range(duration_minutes * 12):  # 12 points per minute (5-second intervals)
        timestamp = start_time + timedelta(seconds=i * 5)
        data_points.append({
            "timestamp": timestamp.isoformat(),
            "cpu_percent": 45.0 + (i % 20),  # Simulated CPU usage
            "memory_percent": 60.0 + (i % 15),  # Simulated memory usage
            "frame_rate": 30.0 - (i % 5),  # Simulated frame rate variation
        })
    
    return data_points

if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"]) 