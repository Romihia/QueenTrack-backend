import pytest
import asyncio
from httpx import AsyncClient
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock, patch
import tempfile
import os
from datetime import datetime
import json
import cv2
import numpy as np

from app.main import app
from app.core.database import db

# Override database settings for testing
TEST_DB_NAME = "queentrack_test"

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
async def test_client():
    """Create a test client for the FastAPI app."""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac

@pytest.fixture
def sync_test_client():
    """Create a synchronous test client for testing WebSocket connections."""
    return TestClient(app)

@pytest.fixture
async def clean_database():
    """Clean the test database before and after each test."""
    # Clean before test
    await db[TEST_DB_NAME].delete_many({})
    yield
    # Clean after test
    await db[TEST_DB_NAME].delete_many({})

@pytest.fixture
def sample_event_data():
    """Sample event data for testing."""
    return {
        "time_out": datetime.now(),
        "time_in": datetime.now(),
        "video_url": "http://example.com/video.mp4"
    }

@pytest.fixture
def sample_event_data_json():
    """Sample event data in JSON format for API calls."""
    return {
        "time_out": datetime.now().isoformat(),
        "time_in": datetime.now().isoformat(),
        "video_url": "http://example.com/video.mp4"
    }

@pytest.fixture
def incomplete_event_data():
    """Incomplete event data for testing partial updates."""
    return {
        "time_out": datetime.now().isoformat(),
        "time_in": None,
        "video_url": None
    }

@pytest.fixture
def mock_video_frame():
    """Create a mock video frame for testing."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Add some pattern to make it more realistic
    cv2.rectangle(frame, (100, 100), (200, 200), (255, 255, 255), -1)
    return frame

@pytest.fixture
def mock_video_bytes():
    """Create mock video bytes for testing."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(frame, (100, 100), (200, 200), (255, 255, 255), -1)
    _, buffer = cv2.imencode('.jpg', frame)
    return buffer.tobytes()

@pytest.fixture
def temp_video_file():
    """Create a temporary video file for testing uploads."""
    # Create a temporary video file
    temp_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
    
    # Create a simple video using OpenCV
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_file.name, fourcc, 20.0, (640, 480))
    
    # Write a few frames
    for i in range(30):  # 1.5 seconds at 20fps
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Add frame number text
        cv2.putText(frame, f'Frame {i}', (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        out.write(frame)
    
    out.release()
    temp_file.close()
    
    yield temp_file.name
    
    # Cleanup
    try:
        os.unlink(temp_file.name)
    except FileNotFoundError:
        pass

@pytest.fixture
def mock_yolo_model():
    """Mock YOLO model for testing."""
    mock_model = MagicMock()
    
    # Mock detection results
    mock_boxes = MagicMock()
    mock_boxes.xyxy = [[100, 100, 200, 200, 0.9, 0]]  # x1, y1, x2, y2, conf, cls
    
    mock_result = MagicMock()
    mock_result.boxes = mock_boxes
    
    mock_model.return_value = [mock_result]
    
    return mock_model

@pytest.fixture
def mock_yolo_classify_model():
    """Mock YOLO classification model for testing."""
    mock_model = MagicMock()
    
    # Mock classification results
    mock_probs = MagicMock()
    mock_probs.top1 = 0  # Class ID for "marked_bee"
    mock_probs.top1conf = 0.85  # High confidence
    
    mock_result = MagicMock()
    mock_result.probs = mock_probs
    mock_result.names = {0: "marked_bee", 1: "normal_bee"}
    
    mock_model.predict.return_value = [mock_result]
    
    return mock_model

@pytest.fixture
def camera_config_data():
    """Sample camera configuration data."""
    return {
        "internal_camera_id": "0",
        "external_camera_id": "1"
    }

@pytest.fixture(autouse=True)
def setup_test_directories():
    """Setup test directories before tests run."""
    test_dirs = [
        "/tmp/test_videos",
        "/tmp/test_videos/outside_videos",
        "/tmp/test_videos/temp_videos",
        "/tmp/test_videos/uploaded_videos",
        "/tmp/test_videos/processed_videos"
    ]
    
    for directory in test_dirs:
        os.makedirs(directory, exist_ok=True)
    
    yield
    
    # Cleanup test directories
    import shutil
    for directory in test_dirs:
        try:
            shutil.rmtree(directory)
        except FileNotFoundError:
            pass

# Mock environment variables for testing
@pytest.fixture(autouse=True)
def mock_env_vars():
    """Mock environment variables for testing."""
    with patch.dict(os.environ, {
        'MONGO_URI': 'mongodb://localhost:27017',
        'MONGO_DB_NAME': TEST_DB_NAME,
        'ENV': 'test'
    }):
        yield 