"""
Timezone service for handling local time adjustments
Adds +3 hours to all server timestamps to match local timezone
"""
import logging
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

class TimezoneService:
    """Service to handle timezone adjustments for the QueenTrack system"""
    
    def __init__(self, timezone_offset_hours: int = 3):
        """
        Initialize timezone service with local offset
        
        Args:
            timezone_offset_hours: Hours to add to server time (default: +3)
        """
        self.timezone_offset = timedelta(hours=timezone_offset_hours)
        try:
            logger.info(f"🕐 TimezoneService initialized with offset: +{timezone_offset_hours} hours")
        except Exception:
            # Fallback if logger not available during startup
            print(f"TimezoneService initialized with offset: +{timezone_offset_hours} hours")
    
    def get_local_now(self) -> datetime:
        """Get current time adjusted for local timezone"""
        return datetime.now() + self.timezone_offset
    
    def to_local_time(self, server_time: datetime) -> datetime:
        """Convert server time to local time"""
        if server_time is None:
            return None
        return server_time + self.timezone_offset
    
    def to_server_time(self, local_time: datetime) -> datetime:
        """Convert local time to server time (if needed)"""
        if local_time is None:
            return None
        return local_time - self.timezone_offset
    
    def format_local_time(self, server_time: Optional[datetime], format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
        """Format server time as local time string"""
        if server_time is None:
            return ""
        
        local_time = self.to_local_time(server_time)
        return local_time.strftime(format_str)
    
    def get_email_timestamp(self, server_time: Optional[datetime] = None) -> datetime:
        """Get timestamp for email notifications in local time"""
        if server_time is None:
            return self.get_local_now()
        return self.to_local_time(server_time)
    
    def update_timezone_offset(self, hours: int):
        """Update the timezone offset"""
        self.timezone_offset = timedelta(hours=hours)
        logger.info(f"🕐 Timezone offset updated to: +{hours} hours")

# Global timezone service instance
timezone_service = TimezoneService()

def get_timezone_service() -> TimezoneService:
    """Get the global timezone service instance"""
    return timezone_service

def get_local_now() -> datetime:
    """Quick access to local time"""
    return timezone_service.get_local_now()

def to_local_time(server_time: datetime) -> datetime:
    """Quick access to convert server time to local time"""
    return timezone_service.to_local_time(server_time)