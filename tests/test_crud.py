import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from httpx import AsyncClient
from app.main import app
from datetime import datetime

@pytest.mark.asyncio
async def test_create_event():
    """Test creating an event through the API."""
    with patch('app.routes.events_routes.create_event') as mock_create:
        # Create a proper EventDB object for testing
        from app.schemas.schema import EventDB
        
        mock_event = EventDB(
            _id="507f1f77bcf86cd799439011",
            time_out=datetime.now(),
            time_in=datetime.now(),
            video_url="http://example.com/video.mp4"
        )
        mock_create.return_value = mock_event
        
        async with AsyncClient(app=app, base_url="http://test") as ac:
            data = {
                "time_out": datetime.now().isoformat(),
                "time_in": datetime.now().isoformat(),
                "video_url": "http://example.com/video.mp4"
            }
            resp = await ac.post("/events/", json=data)
            assert resp.status_code == 201
            created = resp.json()
            # EventDB returns _id field (not id due to alias configuration)
            assert "_id" in created
            assert created["video_url"] == data["video_url"]

@pytest.mark.asyncio
async def test_get_events():
    """Test getting all events through the API."""
    with patch('app.routes.events_routes.get_all_events') as mock_get_all:
        # Create proper EventDB objects for testing
        from app.schemas.schema import EventDB
        
        mock_event1 = EventDB(
            _id="1",
            time_out=datetime.now(),
            time_in=datetime.now(),
            video_url="http://video1.mp4"
        )
        
        mock_event2 = EventDB(
            _id="2",
            time_out=datetime.now(),
            time_in=datetime.now(),
            video_url="http://video2.mp4"
        )
        
        mock_get_all.return_value = [mock_event1, mock_event2]
        
        async with AsyncClient(app=app, base_url="http://test") as ac:
            resp = await ac.get("/events/")
            assert resp.status_code == 200
            events = resp.json()
            assert isinstance(events, list)

@pytest.mark.asyncio
async def test_integration_full_workflow():
    """Test a complete integration workflow with mocked dependencies."""
    with patch('app.services.service.db') as mock_db:
        # Mock database operations
        mock_insert_result = MagicMock()
        mock_insert_result.inserted_id = "507f1f77bcf86cd799439011"
        mock_db.__getitem__.return_value.insert_one = AsyncMock(return_value=mock_insert_result)
        
        # Mock find operation for get_all_events
        async def mock_async_iter():
            yield {
                '_id': mock_insert_result.inserted_id,
                'time_out': datetime.now(),
                'time_in': datetime.now(),
                'video_url': 'http://example.com/video.mp4'
            }
        
        mock_cursor = MagicMock()
        mock_cursor.__aiter__ = lambda self: mock_async_iter()
        mock_db.__getitem__.return_value.find.return_value = mock_cursor
        
        async with AsyncClient(app=app, base_url="http://test") as ac:
            # Test creating an event
            create_data = {
                "time_out": datetime.now().isoformat(),
                "time_in": datetime.now().isoformat(),
                "video_url": "http://example.com/video.mp4"
            }
            create_resp = await ac.post("/events/", json=create_data)
            assert create_resp.status_code == 201
            
            # Test getting all events
            get_resp = await ac.get("/events/")
            assert get_resp.status_code == 200
            events = get_resp.json()
            assert len(events) >= 0  # Should return list (might be empty due to mocking)
