from bson import ObjectId
from typing import List, Optional
from datetime import datetime

from app.core.database import db
from app.schemas.schema import (
    EventCreate, EventUpdate, EventDB,
    SystemSettingsCreate, SystemSettingsUpdate, SystemSettingsDB
)

COLLECTION_NAME = "events"
SETTINGS_COLLECTION_NAME = "system_settings"

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
    try:
        doc = await db[COLLECTION_NAME].find_one({"_id": ObjectId(event_id)})
        if doc:
            doc["_id"] = str(doc["_id"])
            return EventDB(**doc)
        return None
    except Exception:
        return None

async def update_event(event_id: str, update_data: EventUpdate) -> Optional[EventDB]:
    try:
        update_dict = {k: v for k, v in update_data.dict().items() if v is not None}
        if update_dict:
            await db[COLLECTION_NAME].update_one(
                {"_id": ObjectId(event_id)},
                {"$set": update_dict}
            )
        return await get_event_by_id(event_id)
    except Exception:
        return None

async def delete_event(event_id: str) -> bool:
    try:
        result = await db[COLLECTION_NAME].delete_one({"_id": ObjectId(event_id)})
        return result.deleted_count > 0
    except Exception:
        return False

async def get_system_settings() -> Optional[SystemSettingsDB]:
    """קבלת הגדרות המערכת (רק רשומה אחת)"""
    try:
        doc = await db[SETTINGS_COLLECTION_NAME].find_one({})
        if doc:
            doc["_id"] = str(doc["_id"])
            return SystemSettingsDB(**doc)
        return None
    except Exception:
        return None

async def create_or_update_system_settings(settings_data: SystemSettingsCreate) -> SystemSettingsDB:
    """יצירה או עדכון של הגדרות המערכת (רק רשומה אחת)"""
    try:
        # Check if settings already exist
        existing = await db[SETTINGS_COLLECTION_NAME].find_one({})
        
        doc = settings_data.dict()
        doc["updated_at"] = datetime.now()
        
        if existing:
            # Update existing
            doc["created_at"] = existing.get("created_at", datetime.now())
            await db[SETTINGS_COLLECTION_NAME].replace_one(
                {"_id": existing["_id"]}, 
                doc
            )
            doc["_id"] = str(existing["_id"])
        else:
            # Create new
            doc["created_at"] = datetime.now()
            result = await db[SETTINGS_COLLECTION_NAME].insert_one(doc)
            doc["_id"] = str(result.inserted_id)
        
        return SystemSettingsDB(**doc)
    except Exception as e:
        raise Exception(f"Failed to save settings: {str(e)}")

async def delete_system_settings() -> bool:
    """מחיקת הגדרות המערכת"""
    try:
        result = await db[SETTINGS_COLLECTION_NAME].delete_many({})
        return result.deleted_count > 0
    except Exception:
        return False
