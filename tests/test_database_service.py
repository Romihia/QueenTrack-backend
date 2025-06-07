import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from bson import ObjectId
from datetime import datetime

from app.services.service import (
    create_event,
    get_all_events,
    get_event_by_id,
    update_event,
    delete_event
)
from app.schemas.schema import EventCreate, EventUpdate


class TestDatabaseService:
    """Test database service operations with both positive and negative scenarios."""

    @pytest.mark.asyncio
    async def test_create_event_success(self, sample_event_data, clean_database):
        """Test successful event creation."""
        event_create = EventCreate(**sample_event_data)
        
        with patch('app.services.service.db') as mock_db:
            # Mock successful insertion
            mock_result = MagicMock()
            mock_result.inserted_id = ObjectId()
            mock_db.__getitem__.return_value.insert_one = AsyncMock(return_value=mock_result)
            
            result = await create_event(event_create)
            
            assert result is not None
            assert hasattr(result, 'id')
            assert result.time_out == sample_event_data['time_out']
            assert result.time_in == sample_event_data['time_in']
            assert result.video_url == sample_event_data['video_url']

    @pytest.mark.asyncio
    async def test_create_event_database_error(self, sample_event_data):
        """Test event creation with database error."""
        event_create = EventCreate(**sample_event_data)
        
        with patch('app.services.service.db') as mock_db:
            # Mock database error
            mock_db.__getitem__.return_value.insert_one = AsyncMock(side_effect=Exception("Database error"))
            
            with pytest.raises(Exception):
                await create_event(event_create)

    @pytest.mark.asyncio
    async def test_get_all_events_success(self):
        """Test successful retrieval of all events."""
        with patch('app.services.service.db') as mock_db:
            # Mock database response
            sample_doc = {
                '_id': ObjectId(),
                'time_out': datetime.now(),
                'time_in': datetime.now(),
                'video_url': 'http://example.com/video.mp4'
            }
            
            # Create async iterator mock
            async def mock_async_iter():
                yield sample_doc
            
            mock_cursor = MagicMock()
            mock_cursor.__aiter__ = lambda self: mock_async_iter()
            mock_db.__getitem__.return_value.find.return_value = mock_cursor
            
            result = await get_all_events()
            
            assert isinstance(result, list)
            assert len(result) == 1
            assert result[0].video_url == 'http://example.com/video.mp4'

    @pytest.mark.asyncio
    async def test_get_all_events_empty(self):
        """Test retrieval when no events exist."""
        with patch('app.services.service.db') as mock_db:
            # Mock empty cursor
            async def mock_empty_iter():
                return
                yield  # Never reached
            
            mock_cursor = MagicMock()
            mock_cursor.__aiter__ = lambda self: mock_empty_iter()
            mock_db.__getitem__.return_value.find.return_value = mock_cursor
            
            result = await get_all_events()
            
            assert isinstance(result, list)
            assert len(result) == 0

    @pytest.mark.asyncio
    async def test_get_event_by_id_success(self):
        """Test successful event retrieval by ID."""
        event_id = str(ObjectId())
        
        with patch('app.services.service.db') as mock_db:
            # Mock successful find
            sample_doc = {
                '_id': ObjectId(event_id),
                'time_out': datetime.now(),
                'time_in': datetime.now(),
                'video_url': 'http://example.com/video.mp4'
            }
            mock_db.__getitem__.return_value.find_one = AsyncMock(return_value=sample_doc)
            
            result = await get_event_by_id(event_id)
            
            assert result is not None
            assert result.id == event_id
            assert result.video_url == 'http://example.com/video.mp4'

    @pytest.mark.asyncio
    async def test_get_event_by_id_not_found(self):
        """Test event retrieval with non-existent ID."""
        event_id = str(ObjectId())
        
        with patch('app.services.service.db') as mock_db:
            # Mock not found
            mock_db.__getitem__.return_value.find_one = AsyncMock(return_value=None)
            
            result = await get_event_by_id(event_id)
            
            assert result is None

    @pytest.mark.asyncio
    async def test_get_event_by_id_invalid_id(self):
        """Test event retrieval with invalid ObjectId."""
        invalid_id = "invalid_object_id"
        
        result = await get_event_by_id(invalid_id)
        
        assert result is None

    @pytest.mark.asyncio
    async def test_update_event_success(self):
        """Test successful event update."""
        event_id = str(ObjectId())
        update_data = EventUpdate(video_url="http://updated.com/video.mp4")
        
        with patch('app.services.service.db') as mock_db:
            # Mock successful update
            mock_result = MagicMock()
            mock_result.modified_count = 1
            mock_db.__getitem__.return_value.update_one = AsyncMock(return_value=mock_result)
            
            # Mock get_event_by_id for return value
            with patch('app.services.service.get_event_by_id') as mock_get:
                mock_event = MagicMock()
                mock_event.id = event_id
                mock_event.video_url = "http://updated.com/video.mp4"
                mock_get.return_value = mock_event
                
                result = await update_event(event_id, update_data)
                
                assert result is not None
                assert result.video_url == "http://updated.com/video.mp4"

    @pytest.mark.asyncio
    async def test_update_event_not_found(self):
        """Test update of non-existent event."""
        event_id = str(ObjectId())
        update_data = EventUpdate(video_url="http://updated.com/video.mp4")
        
        with patch('app.services.service.db') as mock_db:
            # Mock no modifications (event not found)
            mock_result = MagicMock()
            mock_result.modified_count = 0
            mock_db.__getitem__.return_value.update_one = AsyncMock(return_value=mock_result)
            
            # Mock get_event_by_id returns None
            with patch('app.services.service.get_event_by_id') as mock_get:
                mock_get.return_value = None
                
                result = await update_event(event_id, update_data)
                
                assert result is None

    @pytest.mark.asyncio
    async def test_update_event_invalid_id(self):
        """Test update with invalid ObjectId."""
        invalid_id = "invalid_object_id"
        update_data = EventUpdate(video_url="http://updated.com/video.mp4")
        
        result = await update_event(invalid_id, update_data)
        
        assert result is None

    @pytest.mark.asyncio
    async def test_update_event_no_changes(self):
        """Test update with no actual changes."""
        event_id = str(ObjectId())
        update_data = EventUpdate()  # Empty update
        
        result = await update_event(event_id, update_data)
        
        assert result is None

    @pytest.mark.asyncio
    async def test_delete_event_success(self):
        """Test successful event deletion."""
        event_id = str(ObjectId())
        
        with patch('app.services.service.db') as mock_db:
            # Mock successful deletion
            mock_result = MagicMock()
            mock_result.deleted_count = 1
            mock_db.__getitem__.return_value.delete_one = AsyncMock(return_value=mock_result)
            
            result = await delete_event(event_id)
            
            assert result == 1

    @pytest.mark.asyncio
    async def test_delete_event_not_found(self):
        """Test deletion of non-existent event."""
        event_id = str(ObjectId())
        
        with patch('app.services.service.db') as mock_db:
            # Mock not found
            mock_result = MagicMock()
            mock_result.deleted_count = 0
            mock_db.__getitem__.return_value.delete_one = AsyncMock(return_value=mock_result)
            
            result = await delete_event(event_id)
            
            assert result == 0

    @pytest.mark.asyncio
    async def test_delete_event_invalid_id(self):
        """Test deletion with invalid ObjectId."""
        invalid_id = "invalid_object_id"
        
        result = await delete_event(invalid_id)
        
        assert result == 0

    @pytest.mark.asyncio
    async def test_update_event_partial_data(self):
        """Test partial event update."""
        event_id = str(ObjectId())
        update_data = EventUpdate(video_url="http://updated.com/video.mp4")
        
        with patch('app.services.service.db') as mock_db:
            # Mock successful update
            mock_result = MagicMock()
            mock_result.modified_count = 1
            mock_db.__getitem__.return_value.update_one = AsyncMock(return_value=mock_result)
            
            # Mock get_event_by_id for return value
            with patch('app.services.service.get_event_by_id') as mock_get:
                mock_event = MagicMock()
                mock_event.id = event_id
                mock_event.video_url = "http://updated.com/video.mp4"
                mock_event.time_out = datetime.now()
                mock_event.time_in = datetime.now()
                mock_get.return_value = mock_event
                
                result = await update_event(event_id, update_data)
                
                assert result is not None
                assert result.video_url == "http://updated.com/video.mp4"

    @pytest.mark.asyncio
    async def test_create_event_with_none_values(self):
        """Test creating event with None values."""
        event_data = {
            "time_out": datetime.now(),
            "time_in": datetime.now(),
            "video_url": None
        }
        event_create = EventCreate(**event_data)

        with patch('app.services.service.db') as mock_db:
            # Mock successful insertion
            mock_result = MagicMock()
            mock_result.inserted_id = ObjectId()
            mock_db.__getitem__.return_value.insert_one = AsyncMock(return_value=mock_result)

            result = await create_event(event_create)

            assert result is not None
            assert result.video_url is None
            assert result.time_out == event_data['time_out']
            assert result.time_in == event_data['time_in'] 