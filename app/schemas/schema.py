from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
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
    conversion_status: Optional[str] = "pending"  # pending, processing, completed, failed
    conversion_error: Optional[str] = None

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
    id: Optional[str] = Field(None, alias="_id")  # MongoDB ID

    class Config:
        populate_by_name = True  # Pydantic 2.x
        # allow_population_by_field_name = True  # Pydantic 1.x (old)

# Settings Schemas for MongoDB

class SettingsBase(BaseModel):
    """Base settings schema"""
    processing: Optional[Dict[str, Any]] = None
    camera_config: Optional[Dict[str, Any]] = None

class SettingsCreate(SettingsBase):
    """Schema for creating new settings"""
    pass

class SettingsUpdate(SettingsBase):
    """Schema for updating settings (all fields optional)"""
    pass

class SettingsResponse(SettingsBase):
    """Schema for settings response"""
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    class Config:
        populate_by_name = True

class ProcessingSettingsDB(BaseModel):
    """הגדרות עיבוד לשמירה במונגו"""
    # Detection and Classification Settings
    detection_enabled: bool = True
    classification_enabled: bool = True
    detection_confidence_threshold: float = 0.25
    classification_confidence_threshold: float = 0.3
    computer_vision_fallback: bool = True
    
    # Drawing Settings
    draw_bee_path: bool = True
    draw_center_line: bool = True
    draw_roi_box: bool = True
    draw_status_text: bool = True
    draw_confidence_scores: bool = True
    draw_timestamp: bool = True
    draw_frame_counter: bool = False
    
    # Path and Tracking Settings
    path_history_size: int = 50
    min_consecutive_detections: int = 3
    transition_cooldown: float = 2.0
    
    # Computer Vision Settings
    cv_contour_min_area: int = 50
    cv_contour_max_area: int = 2000
    cv_aspect_ratio_min: float = 0.3
    cv_aspect_ratio_max: float = 3.0
    
    # Video Management Settings
    auto_delete_videos_after_session: bool = False
    keep_original_videos: bool = True
    auto_convert_videos: bool = True
    video_buffer_seconds: int = 5
    
    # ROI Settings
    roi_x_min: int = 640
    roi_y_min: int = 0
    roi_x_max: int = 1280
    roi_y_max: int = 720
    
    # Email Notification Settings
    email_notifications_enabled: bool = True
    email_on_exit: bool = True
    email_on_entrance: bool = True
    notification_email: Optional[str] = None

class CameraSettings(BaseModel):
    """הגדרות מצלמות"""
    internal_camera_id: str = "0"
    external_camera_id: str = "1"

class SystemSettingsCreate(BaseModel):
    """ליצירת הגדרת מערכת חדשה"""
    processing: ProcessingSettingsDB = ProcessingSettingsDB()
    camera: CameraSettings = CameraSettings()

class SystemSettingsUpdate(BaseModel):
    """לעדכון הגדרת מערכת"""
    processing: Optional[ProcessingSettingsDB] = None
    camera: Optional[CameraSettings] = None

class SystemSettingsDB(BaseModel):
    """הגדרת מערכת כפי שהיא נשמרת במונגו"""
    id: Optional[str] = Field(None, alias="_id")
    processing: ProcessingSettingsDB
    camera: CameraSettings
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    class Config:
        populate_by_name = True

# Notifications Schemas for MongoDB

class NotificationBase(BaseModel):
    """התרעה בסיסית"""
    event_type: str  # 'exit' או 'entrance'
    message: str
    timestamp: datetime
    read: bool = False
    additional_data: Optional[Dict[str, Any]] = None

class NotificationCreate(NotificationBase):
    """ליצירת התרעה חדשה"""
    pass

class NotificationUpdate(BaseModel):
    """לעדכון התרעה"""
    read: Optional[bool] = None

class NotificationDB(NotificationBase):
    """התרעה כפי שהיא נשמרת במונגו"""
    id: Optional[str] = Field(None, alias="_id")
    created_at: Optional[datetime] = None

    class Config:
        populate_by_name = True
