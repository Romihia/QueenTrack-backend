import pytest
from httpx import AsyncClient
from app.main import app
from datetime import datetime

@pytest.mark.asyncio
async def test_create_event():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        data = {
            "time_out": datetime.now().isoformat(),
            "time_in": datetime.now().isoformat(),
            "video_url": "http://example.com/video.mp4"
        }
        resp = await ac.post("/events/", json=data)
        assert resp.status_code == 201
        created = resp.json()
        assert "id" in created
        assert created["video_url"] == data["video_url"]

@pytest.mark.asyncio
async def test_get_events():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        resp = await ac.get("/events/")
        assert resp.status_code == 200
        events = resp.json()
        assert isinstance(events, list)
