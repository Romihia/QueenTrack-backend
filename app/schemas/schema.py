from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

"""
סכמות Pydantic לייצוג בקשות/תגובות עבור Event:
 - EventCreate: מה נדרש כדי ליצור אירוע
 - EventUpdate: מה נדרש כדי לעדכן אירוע
 - EventDB: איך אירוע מוחזר ללקוח (כולל id)
"""

class EventBase(BaseModel):
    time_out: datetime
    time_in: Optional[datetime] = None
    internal_video_url: Optional[str] = None  # וידאו מהמצלמה הפנימית (מקורי)
    external_video_url: Optional[str] = None  # וידאו מהמצלמה החיצונית (מקורי)
    
    # Converted videos (avc1 format for browser compatibility)
    internal_video_url_converted: Optional[str] = None  # וידאו פנימי מומר (avc1)
    external_video_url_converted: Optional[str] = None  # וידאו חיצוני מומר (avc1)
    
    # Video processing status
    conversion_status: Optional[str] = None  # "pending", "processing", "completed", "failed"
    conversion_error: Optional[str] = None  # הודעת שגיאה במקרה של כשל
    
    # Backward compatibility 
    video_url: Optional[str] = None  # שדה ישן לתאימות אחורית

class EventCreate(EventBase):
    """סכמה ליצירת אירוע חדש (כוללת time_out, time_in, video_url)"""
    pass

class EventUpdate(BaseModel):
    """סכמה לעדכון אירוע (כל השדות אופציונליים)"""
    time_out: Optional[datetime] = None
    time_in: Optional[datetime] = None
    internal_video_url: Optional[str] = None
    external_video_url: Optional[str] = None
    internal_video_url_converted: Optional[str] = None
    external_video_url_converted: Optional[str] = None
    conversion_status: Optional[str] = None
    conversion_error: Optional[str] = None
    video_url: Optional[str] = None  # שדה ישן לתאימות אחורית

class EventDB(EventBase):
    """סכמה שמייצגת את האירוע כפי שנשמר/מוחזר מה-DB"""
    id: str = Field(..., alias="_id")  

    model_config = {
        "populate_by_name": True,  # Updated from 'allow_population_by_field_name' for Pydantic V2
        "from_attributes": True
    }
