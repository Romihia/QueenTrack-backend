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

async def get_filtered_events(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    event_type: Optional[str] = None,
    limit: int = 100,
    skip: int = 0,
    sort_by: str = "time_out",
    sort_order: str = "desc"
) -> List[EventDB]:
    """
    Get events with filtering, pagination, and sorting capabilities
    
    Args:
        start_date: Filter events from this date
        end_date: Filter events until this date  
        event_type: Filter by event type (exit=only exits, entrance=only returns, both=all)
        limit: Maximum number of events to return
        skip: Number of events to skip for pagination
        sort_by: Field to sort by (time_out, time_in)
        sort_order: Sort order (asc/desc)
    
    Returns:
        List of filtered EventDB objects
    """
    try:
        # Build MongoDB query filter
        query_filter = {}
        
        # Date range filtering
        date_conditions = []
        if start_date:
            date_conditions.append({"time_out": {"$gte": start_date}})
        if end_date:
            date_conditions.append({"time_out": {"$lte": end_date}})
        
        if date_conditions:
            if len(date_conditions) == 1:
                query_filter.update(date_conditions[0])
            else:
                query_filter["$and"] = date_conditions
        
        # Event type filtering
        if event_type == "exit":
            # Only events with time_out but no time_in (bee left but didn't return)
            query_filter["time_in"] = {"$exists": False}
        elif event_type == "entrance":
            # Only events with both time_out and time_in (bee returned)
            query_filter["time_in"] = {"$exists": True, "$ne": None}
        # For "both" or None, no additional filter needed
        
        # Build sort parameters
        sort_direction = 1 if sort_order == "asc" else -1
        sort_params = [(sort_by, sort_direction)]
        
        # Execute query with filtering, sorting, and pagination
        cursor = db[COLLECTION_NAME].find(query_filter).sort(sort_params).skip(skip).limit(limit)
        
        events = []
        async for doc in cursor:
            doc["_id"] = str(doc["_id"])
            events.append(EventDB(**doc))
        
        return events
        
    except Exception as e:
        print(f"Error in get_filtered_events: {e}")
        # Fallback to getting all events if filtering fails
        return await get_all_events()

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

async def get_events_statistics(start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> dict:
    """
    Get comprehensive event statistics for the specified date range
    
    Args:
        start_date: Start date for statistics (optional)
        end_date: End date for statistics (optional)
    
    Returns:
        Dictionary containing various statistics about events
    """
    try:
        # Build date filter
        date_filter = {}
        if start_date or end_date:
            time_out_filter = {}
            if start_date:
                time_out_filter["$gte"] = start_date
            if end_date:
                time_out_filter["$lte"] = end_date
            date_filter["time_out"] = time_out_filter
        
        # Get total events count
        total_events = await db[COLLECTION_NAME].count_documents(date_filter)
        
        # Get events with time_in (bees that returned)
        returned_filter = {**date_filter, "time_in": {"$exists": True, "$ne": None}}
        returned_events = await db[COLLECTION_NAME].count_documents(returned_filter)
        
        # Get events without time_in (bees that left but didn't return yet)
        exit_only_filter = {**date_filter, "time_in": {"$exists": False}}
        exit_only_events = await db[COLLECTION_NAME].count_documents(exit_only_filter)
        
        # Get events by day (for activity patterns)
        pipeline = [
            {"$match": date_filter},
            {
                "$group": {
                    "_id": {
                        "year": {"$year": "$time_out"},
                        "month": {"$month": "$time_out"},
                        "day": {"$dayOfMonth": "$time_out"}
                    },
                    "count": {"$sum": 1}
                }
            },
            {"$sort": {"_id.year": 1, "_id.month": 1, "_id.day": 1}}
        ]
        
        daily_activity = []
        async for doc in db[COLLECTION_NAME].aggregate(pipeline):
            date_str = f"{doc['_id']['year']}-{doc['_id']['month']:02d}-{doc['_id']['day']:02d}"
            daily_activity.append({
                "date": date_str,
                "count": doc["count"]
            })
        
        # Get events by hour (for hourly patterns)
        hourly_pipeline = [
            {"$match": date_filter},
            {
                "$group": {
                    "_id": {"$hour": "$time_out"},
                    "count": {"$sum": 1}
                }
            },
            {"$sort": {"_id": 1}}
        ]
        
        hourly_activity = []
        async for doc in db[COLLECTION_NAME].aggregate(hourly_pipeline):
            hourly_activity.append({
                "hour": doc["_id"],
                "count": doc["count"]
            })
        
        # Calculate average session duration for returned bees
        duration_pipeline = [
            {"$match": returned_filter},
            {
                "$project": {
                    "duration": {
                        "$subtract": ["$time_in", "$time_out"]
                    }
                }
            },
            {
                "$group": {
                    "_id": None,
                    "avg_duration": {"$avg": "$duration"},
                    "min_duration": {"$min": "$duration"},
                    "max_duration": {"$max": "$duration"}
                }
            }
        ]
        
        duration_stats = None
        async for doc in db[COLLECTION_NAME].aggregate(duration_pipeline):
            duration_stats = {
                "average_seconds": int(doc["avg_duration"] / 1000) if doc["avg_duration"] else 0,
                "min_seconds": int(doc["min_duration"] / 1000) if doc["min_duration"] else 0,
                "max_seconds": int(doc["max_duration"] / 1000) if doc["max_duration"] else 0
            }
        
        return {
            "total_events": total_events,
            "returned_events": returned_events, 
            "exit_only_events": exit_only_events,
            "return_rate": round((returned_events / total_events * 100), 2) if total_events > 0 else 0,
            "daily_activity": daily_activity,
            "hourly_activity": hourly_activity,
            "session_duration": duration_stats,
            "date_range": {
                "start_date": start_date.isoformat() if start_date else None,
                "end_date": end_date.isoformat() if end_date else None
            }
        }
        
    except Exception as e:
        print(f"Error in get_events_statistics: {e}")
        return {
            "total_events": 0,
            "returned_events": 0,
            "exit_only_events": 0,
            "return_rate": 0,
            "daily_activity": [],
            "hourly_activity": [],
            "session_duration": None,
            "date_range": {
                "start_date": start_date.isoformat() if start_date else None,
                "end_date": end_date.isoformat() if end_date else None
            },
            "error": str(e)
        }

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
