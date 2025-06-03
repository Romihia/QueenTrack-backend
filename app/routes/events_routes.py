from fastapi import APIRouter, HTTPException, status
from typing import List
from app.schemas.schema import EventCreate, EventDB, EventUpdate
from app.services.service import (
    create_event,
    get_event_by_id,
    get_all_events,
    update_event,
    delete_event
)

router = APIRouter()

@router.post("/", response_model=EventDB, status_code=status.HTTP_201_CREATED)
async def create_event_route(event_data: EventCreate):
    new_event = await create_event(event_data)
    return new_event

@router.get("/", response_model=List[EventDB])
async def get_all_events_route():
    return await get_all_events()

@router.get("/{event_id}", response_model=EventDB)
async def get_event_route(event_id: str):
    ev = await get_event_by_id(event_id)
    if not ev:
        raise HTTPException(status_code=404, detail="Event not found")
    return ev

@router.put("/{event_id}", response_model=EventDB)
async def update_event_route(event_id: str, event_data: EventUpdate):
    updated = await update_event(event_id, event_data)
    if not updated:
        raise HTTPException(status_code=404, detail="Event not found or no changes")
    return updated

@router.delete("/{event_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_event_route(event_id: str):
    deleted_count = await delete_event(event_id)
    if deleted_count == 0:
        raise HTTPException(status_code=404, detail="Event not found")
    return None
