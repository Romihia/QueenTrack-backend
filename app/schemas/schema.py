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
    internal_video_url: Optional[str] = None  # וידאו מהמצלמה הפנימית
    external_video_url: Optional[str] = None  # וידאו מהמצלמה החיצונית
    
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
    video_url: Optional[str] = None  # שדה ישן לתאימות אחורית

class EventDB(EventBase):
    """סכמה שמייצגת את האירוע כפי שנשמר/מוחזר מה-DB"""
    id: str = Field(..., alias="_id")  

    class Config:
        allow_population_by_field_name = True
        # כדי לתמוך בהמרת _id -> id
        from_attributes = True
