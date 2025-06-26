from bson import ObjectId
from typing import List, Optional
from datetime import datetime

from app.core.database import db
from app.schemas.schema import (
    EventCreate, EventUpdate, EventDB,
    SystemSettingsCreate, SystemSettingsUpdate, SystemSettingsDB,
    NotificationCreate, NotificationUpdate, NotificationDB
)

COLLECTION_NAME = "events"
SETTINGS_COLLECTION_NAME = "system_settings"
NOTIFICATIONS_COLLECTION_NAME = "notifications"

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

# Notifications functions
async def create_notification(notification_data: NotificationCreate) -> NotificationDB:
    """יצירת התרעה חדשה"""
    try:
        doc = notification_data.dict()
        doc["created_at"] = datetime.now()
        result = await db[NOTIFICATIONS_COLLECTION_NAME].insert_one(doc)
        doc["_id"] = str(result.inserted_id)
        return NotificationDB(**doc)
    except Exception as e:
        raise Exception(f"Failed to create notification: {str(e)}")

async def get_all_notifications(limit: int = 50) -> List[NotificationDB]:
    """קבלת כל ההתרעות (מוגבל ל-50 אחרונות)"""
    try:
        cursor = db[NOTIFICATIONS_COLLECTION_NAME].find({}).sort("timestamp", -1).limit(limit)
        notifications = []
        async for doc in cursor:
            doc["_id"] = str(doc["_id"])
            notifications.append(NotificationDB(**doc))
        return notifications
    except Exception:
        return []

async def get_unread_notifications_count() -> int:
    """קבלת מספר ההתרעות שלא נקראו"""
    try:
        count = await db[NOTIFICATIONS_COLLECTION_NAME].count_documents({"read": False})
        return count
    except Exception:
        return 0

async def mark_all_notifications_read() -> bool:
    """סימון כל ההתרעות כנקראו"""
    try:
        result = await db[NOTIFICATIONS_COLLECTION_NAME].update_many(
            {"read": False},
            {"$set": {"read": True}}
        )
        return True
    except Exception:
        return False

async def delete_all_notifications() -> bool:
    """מחיקת כל ההתרעות"""
    try:
        result = await db[NOTIFICATIONS_COLLECTION_NAME].delete_many({})
        return result.deleted_count > 0
    except Exception:
        return False

async def delete_notification(notification_id: str) -> bool:
    """מחיקת התרעה ספציפית"""
    try:
        result = await db[NOTIFICATIONS_COLLECTION_NAME].delete_one({"_id": ObjectId(notification_id)})
        return result.deleted_count > 0
    except Exception:
        return False

# Enhanced event functions
async def delete_event_with_videos(event_id: str) -> bool:
    """מחיקת אירוע כולל הווידאו הקשור אליו"""
    try:
        import os
        import glob
        
        # Get event details first
        event = await get_event_by_id(event_id)
        if not event:
            return False
        
        # Collect all video paths to delete
        video_paths_to_delete = []
        
        # Original videos
        if event.internal_video_url:
            # Convert URL to file path
            video_path = event.internal_video_url.replace("/videos/", "/data/videos/")
            if os.path.exists(video_path):
                video_paths_to_delete.append(video_path)
        
        if event.external_video_url:
            video_path = event.external_video_url.replace("/videos/", "/data/videos/")
            if os.path.exists(video_path):
                video_paths_to_delete.append(video_path)
        
        # Converted videos
        if event.internal_video_url_converted:
            video_path = event.internal_video_url_converted.replace("/videos/", "/data/videos/")
            if os.path.exists(video_path):
                video_paths_to_delete.append(video_path)
        
        if event.external_video_url_converted:
            video_path = event.external_video_url_converted.replace("/videos/", "/data/videos/")
            if os.path.exists(video_path):
                video_paths_to_delete.append(video_path)
        
        # Try to find event directory and delete all videos in it
        event_dir_pattern = f"/data/videos/events/*{event_id}*"
        event_directories = glob.glob(event_dir_pattern)
        
        for event_dir in event_directories:
            if os.path.exists(event_dir):
                video_files = glob.glob(os.path.join(event_dir, "*.mp4"))
                video_paths_to_delete.extend(video_files)
        
        # Delete all video files
        deleted_videos = []
        for video_path in video_paths_to_delete:
            try:
                if os.path.exists(video_path):
                    os.remove(video_path)
                    deleted_videos.append(video_path)
            except Exception as e:
                print(f"Failed to delete video {video_path}: {e}")
        
        # Delete event directories if empty
        for event_dir in event_directories:
            try:
                if os.path.exists(event_dir) and not os.listdir(event_dir):
                    os.rmdir(event_dir)
            except Exception as e:
                print(f"Failed to delete directory {event_dir}: {e}")
        
        # Delete from database
        result = await db[COLLECTION_NAME].delete_one({"_id": ObjectId(event_id)})
        
        if result.deleted_count > 0:
            print(f"Event {event_id} deleted successfully. Deleted videos: {deleted_videos}")
            return True
        else:
            return False
            
    except Exception as e:
        print(f"Error deleting event {event_id}: {e}")
        return False
