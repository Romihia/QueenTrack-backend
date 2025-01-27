from bson import ObjectId
from typing import List, Optional
from datetime import datetime

from app.core.database import db
from app.schemas.schema import EventCreate, EventUpdate, EventDB

COLLECTION_NAME = "events"

async def create_event(event_data: EventCreate) -> EventDB:
    doc = event_data.dict()
    # doc = {"time_out": datetime(...), "time_in": datetime(...), "video_url": ...}
    result = await db[COLLECTION_NAME].insert_one(doc)
    doc["_id"] = str(result.inserted_id)
    return EventDB(**doc)

async def get_all_events() -> List[EventDB]:
    cursor = db[COLLECTION_NAME].find({})
    events = []
    async for doc in cursor:
        doc["_id"] = str(doc["_id"])
        events.append(EventDB(**doc))
    return events

async def get_event_by_id(event_id: str) -> Optional[EventDB]:
    if not ObjectId.is_valid(event_id):
        return None
    raw_doc = await db[COLLECTION_NAME].find_one({"_id": ObjectId(event_id)})
    if raw_doc:
        raw_doc["_id"] = str(raw_doc["_id"])
        return EventDB(**raw_doc)
    return None

async def update_event(event_id: str, event_data: EventUpdate) -> Optional[EventDB]:
    if not ObjectId.is_valid(event_id):
        return None
    to_update = {k: v for k, v in event_data.dict(exclude_unset=True).items() if v is not None}
    if not to_update:
        return None

    result = await db[COLLECTION_NAME].update_one(
        {"_id": ObjectId(event_id)},
        {"$set": to_update}
    )
    if result.modified_count == 0:
        # ייתכן גם מקרה שהמסמך לא נמצא, או שלא בוצע שינוי
        # נבדוק אם המסמך קיים
        existing = await get_event_by_id(event_id)
        if not existing:
            return None
        # ייתכן שהיה זהה לערכים הקיימים - במקרה זה נחזיר את הקיים
        return existing

    return await get_event_by_id(event_id)

async def delete_event(event_id: str) -> int:
    if not ObjectId.is_valid(event_id):
        return 0
    result = await db[COLLECTION_NAME].delete_one({"_id": ObjectId(event_id)})
    return result.deleted_count
