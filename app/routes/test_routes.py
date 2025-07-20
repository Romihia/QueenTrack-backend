"""
Test Routes - End-to-end testing functionality for camera streaming and external trigger validation
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import asyncio
import json
import os
import time
from datetime import datetime, timedelta
import logging
import subprocess
import threading
from contextlib import contextmanager

# Setup logging
logger = logging.getLogger(__name__)

router = APIRouter()

class TestConfiguration(BaseModel):
    test_name: str = "dual_camera_stream_test"
    test_duration: int = 60  # seconds
    video_file_path: Optional[str] = None
    enable_external_trigger: bool = True
    log_level: str = "DEBUG"
    capture_docker_logs: bool = True
    capture_performance_metrics: bool = True

class TestStatus(BaseModel):
    test_id: str
    status: str  # "pending", "running", "completed", "failed"
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    logs_file: Optional[str] = None
    metrics: Dict[str, Any] = {}

# Global test storage
active_tests: Dict[str, TestStatus] = {}
test_logs: Dict[str, List[str]] = {}

class TestLogCapture:
    """Context manager for capturing all logs during a test"""
    
    def __init__(self, test_id: str, capture_docker: bool = True):
        self.test_id = test_id
        self.capture_docker = capture_docker
        self.log_entries = []
        self.start_time = datetime.now()
        self.docker_log_process = None
        
    def __enter__(self):
        # Initialize log capture
        test_logs[self.test_id] = []
        
        # Start Docker log capture if enabled
        if self.capture_docker:
            self._start_docker_log_capture()
            
        # Add custom log handler
        self._add_test_log_handler()
        
        logger.info(f"🧪 [TEST {self.test_id}] Log capture started at {self.start_time}")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Stop Docker log capture
        if self.docker_log_process:
            self.docker_log_process.terminate()
            
        # Remove custom log handler
        self._remove_test_log_handler()
        
        # Save logs to file
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        
        log_file_path = self._save_logs_to_file(duration)
        
        # Update test status
        if self.test_id in active_tests:
            active_tests[self.test_id].logs_file = log_file_path
            active_tests[self.test_id].end_time = end_time
            
        logger.info(f"🧪 [TEST {self.test_id}] Log capture completed. Duration: {duration:.2f}s")
        logger.info(f"🧪 [TEST {self.test_id}] Logs saved to: {log_file_path}")
        
    def _start_docker_log_capture(self):
        """Start capturing Docker container logs"""
        try:
            # Start docker logs process in background
            def capture_docker_logs():
                try:
                    process = subprocess.Popen([
                        "docker", "logs", "-f", "--timestamps", 
                        "queentrack_backend_prod"
                    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                    text=True, bufsize=1, universal_newlines=True)
                    
                    for line in process.stdout:
                        if line.strip():
                            timestamp = datetime.now().isoformat()
                            log_entry = f"[DOCKER {timestamp}] {line.strip()}"
                            test_logs[self.test_id].append(log_entry)
                            
                except Exception as e:
                    logger.error(f"Error capturing Docker logs: {e}")
                    
            docker_thread = threading.Thread(target=capture_docker_logs, daemon=True)
            docker_thread.start()
            
        except Exception as e:
            logger.error(f"Failed to start Docker log capture: {e}")
            
    def _add_test_log_handler(self):
        """Add a custom log handler for the test"""
        class TestLogHandler(logging.Handler):
            def __init__(self, test_id):
                super().__init__()
                self.test_id = test_id
                
            def emit(self, record):
                try:
                    log_message = self.format(record)
                    timestamp = datetime.now().isoformat()
                    formatted_entry = f"[BACKEND {timestamp}] {log_message}"
                    
                    if self.test_id in test_logs:
                        test_logs[self.test_id].append(formatted_entry)
                except:
                    pass
                    
        # Add handler to root logger
        self.test_handler = TestLogHandler(self.test_id)
        self.test_handler.setLevel(logging.DEBUG)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.test_handler.setFormatter(formatter)
        
        root_logger = logging.getLogger()
        root_logger.addHandler(self.test_handler)
        
    def _remove_test_log_handler(self):
        """Remove the custom log handler"""
        try:
            root_logger = logging.getLogger()
            root_logger.removeHandler(self.test_handler)
        except:
            pass
            
    def _save_logs_to_file(self, duration: float) -> str:
        """Save captured logs to a file"""
        try:
            # Create logs directory
            log_dir = "/data/test_logs"
            os.makedirs(log_dir, exist_ok=True)
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_filename = f"test_{self.test_id}_{timestamp}.log"
            log_file_path = os.path.join(log_dir, log_filename)
            
            # Prepare log content
            log_content = []
            log_content.append(f"=== END-TO-END TEST LOG ===")
            log_content.append(f"Test ID: {self.test_id}")
            log_content.append(f"Start Time: {self.start_time}")
            log_content.append(f"Duration: {duration:.2f} seconds")
            log_content.append(f"Total Log Entries: {len(test_logs.get(self.test_id, []))}")
            log_content.append("=" * 50)
            log_content.append("")
            
            # Add all captured logs
            if self.test_id in test_logs:
                log_content.extend(test_logs[self.test_id])
            
            # Add system information
            log_content.append("")
            log_content.append("=== SYSTEM INFORMATION ===")
            log_content.append(f"Docker Container: queentrack_backend_prod")
            log_content.append(f"Backend Path: D:\\Studies\\semester 7\\פרוייקט גמר\\QueenTrack-backend")
            log_content.append(f"Frontend Path: D:\\Studies\\semester 7\\פרוייקט גמר\\queen-track-frontend")
            
            # Write to file
            with open(log_file_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(log_content))
                
            return log_file_path
            
        except Exception as e:
            logger.error(f"Failed to save logs to file: {e}")
            return None

@router.post("/end-to-end", response_model=Dict[str, Any])
async def start_end_to_end_test(config: TestConfiguration, background_tasks: BackgroundTasks):
    """
    Start an end-to-end test of the dual camera streaming system
    """
    # Generate test ID
    test_id = f"{config.test_name}_{int(time.time())}"
    
    # Create test status
    test_status = TestStatus(
        test_id=test_id,
        status="pending",
        start_time=datetime.now()
    )
    
    active_tests[test_id] = test_status
    
    # Start test in background
    background_tasks.add_task(run_end_to_end_test, test_id, config)
    
    logger.info(f"🧪 [TEST {test_id}] End-to-end test initiated")
    logger.info(f"🧪 [TEST {test_id}] Configuration: {config.dict()}")
    
    return {
        "test_id": test_id,
        "status": "initiated",
        "message": "End-to-end test started",
        "estimated_duration": config.test_duration,
        "log_capture_enabled": config.capture_docker_logs
    }

async def run_end_to_end_test(test_id: str, config: TestConfiguration):
    """
    Execute the end-to-end test sequence
    """
    try:
        # Update status to running
        active_tests[test_id].status = "running"
        
        with TestLogCapture(test_id, config.capture_docker_logs) as log_capture:
            
            logger.info(f"🧪 [TEST {test_id}] Starting end-to-end test sequence")
            logger.info(f"🧪 [TEST {test_id}] Test Duration: {config.test_duration} seconds")
            logger.info(f"🧪 [TEST {test_id}] External Trigger Enabled: {config.enable_external_trigger}")
            
            # Phase 1: System Health Check
            logger.info(f"🧪 [TEST {test_id}] PHASE 1: System Health Check")
            await check_system_health(test_id)
            
            # Phase 2: WebSocket Connection Test
            logger.info(f"🧪 [TEST {test_id}] PHASE 2: WebSocket Connection Test")
            await test_websocket_connections(test_id)
            
            # Phase 3: Camera Detection Test
            logger.info(f"🧪 [TEST {test_id}] PHASE 3: Camera Detection Test")
            await test_camera_detection(test_id)
            
            # Phase 4: External Camera Trigger Test
            if config.enable_external_trigger:
                logger.info(f"🧪 [TEST {test_id}] PHASE 4: External Camera Trigger Test")
                await test_external_camera_trigger(test_id)
            
            # Phase 5: Performance Monitoring
            if config.capture_performance_metrics:
                logger.info(f"🧪 [TEST {test_id}] PHASE 5: Performance Monitoring")
                await monitor_performance_metrics(test_id, config.test_duration)
            
            # Wait for test duration
            logger.info(f"🧪 [TEST {test_id}] PHASE 6: Monitoring for {config.test_duration} seconds")
            await asyncio.sleep(config.test_duration)
            
        # Update status to completed
        active_tests[test_id].status = "completed"
        logger.info(f"🧪 [TEST {test_id}] End-to-end test completed successfully")
        
    except Exception as e:
        # Update status to failed
        active_tests[test_id].status = "failed"
        logger.error(f"🧪 [TEST {test_id}] End-to-end test failed: {e}")

async def check_system_health(test_id: str):
    """Check system health and log status"""
    try:
        # Check Docker container status
        result = subprocess.run([
            "docker", "ps", "--filter", "name=queentrack_backend_prod", "--format", "{{.Status}}"
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0 and "Up" in result.stdout:
            logger.info(f"🧪 [TEST {test_id}] ✅ Docker container is running: {result.stdout.strip()}")
        else:
            logger.error(f"🧪 [TEST {test_id}] ❌ Docker container not running properly")
            
        # Check container health
        health_result = subprocess.run([
            "docker", "inspect", "queentrack_backend_prod", "--format", "{{.State.Health.Status}}"
        ], capture_output=True, text=True, timeout=10)
        
        if health_result.returncode == 0:
            health_status = health_result.stdout.strip()
            logger.info(f"🧪 [TEST {test_id}] Container health status: {health_status}")
            
    except Exception as e:
        logger.error(f"🧪 [TEST {test_id}] System health check failed: {e}")

async def test_websocket_connections(test_id: str):
    """Test WebSocket connection establishment"""
    try:
        import websockets
        
        # Test main live-stream endpoint
        uri = "ws://localhost:8000/video/live-stream"
        logger.info(f"🧪 [TEST {test_id}] Testing WebSocket connection to {uri}")
        
        try:
            async with websockets.connect(uri, timeout=10) as websocket:
                logger.info(f"🧪 [TEST {test_id}] ✅ WebSocket connection established")
                
                # Wait for initial message
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                    logger.info(f"🧪 [TEST {test_id}] ✅ Received initial message: {message}")
                except asyncio.TimeoutError:
                    logger.warning(f"🧪 [TEST {test_id}] ⚠️ No initial message received")
                    
        except Exception as conn_error:
            logger.error(f"🧪 [TEST {test_id}] ❌ WebSocket connection failed: {conn_error}")
            
    except ImportError:
        logger.error(f"🧪 [TEST {test_id}] ❌ websockets library not available")

async def test_camera_detection(test_id: str):
    """Test camera detection functionality"""
    try:
        logger.info(f"🧪 [TEST {test_id}] Checking available cameras...")
        
        # This would normally check for actual cameras
        # For now, we'll log that we're testing camera detection
        logger.info(f"🧪 [TEST {test_id}] Camera detection test placeholder")
        logger.info(f"🧪 [TEST {test_id}] Expected cameras: Internal (webcam), External (USB)")
        
    except Exception as e:
        logger.error(f"🧪 [TEST {test_id}] Camera detection test failed: {e}")

async def test_external_camera_trigger(test_id: str):
    """Test external camera trigger functionality"""
    try:
        logger.info(f"🧪 [TEST {test_id}] Testing external camera trigger mechanism")
        
        # Import the external camera activation function
        from app.routes.video_routes import send_external_camera_activation
        
        # Simulate a trigger event
        logger.info(f"🧪 [TEST {test_id}] Simulating bee exit event trigger")
        
        # Call the external camera activation
        trigger_result = await send_external_camera_activation("test_event_123", test_id)
        
        if trigger_result:
            logger.info(f"🧪 [TEST {test_id}] ✅ External camera trigger sent successfully")
        else:
            logger.error(f"🧪 [TEST {test_id}] ❌ External camera trigger failed")
            
    except Exception as e:
        logger.error(f"🧪 [TEST {test_id}] External camera trigger test failed: {e}")

async def monitor_performance_metrics(test_id: str, duration: int):
    """Monitor system performance during test"""
    try:
        logger.info(f"🧪 [TEST {test_id}] Starting performance monitoring for {duration}s")
        
        # Get initial metrics
        start_metrics = await get_system_metrics()
        logger.info(f"🧪 [TEST {test_id}] Initial metrics: {start_metrics}")
        
        # Monitor periodically
        monitoring_interval = min(10, duration // 4)  # Monitor every 10s or 4 times during test
        
        for i in range(0, duration, monitoring_interval):
            await asyncio.sleep(monitoring_interval)
            
            current_metrics = await get_system_metrics()
            logger.info(f"🧪 [TEST {test_id}] Metrics at {i+monitoring_interval}s: {current_metrics}")
            
    except Exception as e:
        logger.error(f"🧪 [TEST {test_id}] Performance monitoring failed: {e}")

async def get_system_metrics() -> Dict[str, Any]:
    """Get current system metrics"""
    try:
        # Get Docker container stats
        result = subprocess.run([
            "docker", "stats", "queentrack_backend_prod", "--no-stream", "--format",
            "{{.CPUPerc}},{{.MemUsage}},{{.NetIO}}"
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            stats = result.stdout.strip().split(',')
            return {
                "cpu_percent": stats[0] if len(stats) > 0 else "N/A",
                "memory_usage": stats[1] if len(stats) > 1 else "N/A", 
                "network_io": stats[2] if len(stats) > 2 else "N/A",
                "timestamp": datetime.now().isoformat()
            }
        else:
            return {"error": "Failed to get Docker stats"}
            
    except Exception as e:
        return {"error": str(e)}

@router.get("/status/{test_id}", response_model=TestStatus)
async def get_test_status(test_id: str):
    """Get the status of a running test"""
    if test_id not in active_tests:
        raise HTTPException(status_code=404, detail="Test not found")
    
    return active_tests[test_id]

@router.get("/logs/{test_id}")
async def get_test_logs(test_id: str):
    """Get real-time logs for a test"""
    if test_id not in test_logs:
        raise HTTPException(status_code=404, detail="Test logs not found")
    
    return {
        "test_id": test_id,
        "log_count": len(test_logs[test_id]),
        "logs": test_logs[test_id][-50:]  # Return last 50 log entries
    }

@router.get("/download-logs/{test_id}")
async def download_test_logs(test_id: str):
    """Download complete test logs as a file"""
    if test_id not in active_tests:
        raise HTTPException(status_code=404, detail="Test not found")
    
    test_status = active_tests[test_id]
    
    if not test_status.logs_file or not os.path.exists(test_status.logs_file):
        raise HTTPException(status_code=404, detail="Log file not found")
    
    from fastapi.responses import FileResponse
    
    return FileResponse(
        path=test_status.logs_file,
        filename=f"test_{test_id}_logs.log",
        media_type="text/plain"
    )

@router.get("/active-tests")
async def get_active_tests():
    """Get list of all active tests"""
    return {
        "active_tests": list(active_tests.keys()),
        "test_statuses": {test_id: status.dict() for test_id, status in active_tests.items()}
    }