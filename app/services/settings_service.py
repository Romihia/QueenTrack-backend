import logging
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from motor.motor_asyncio import AsyncIOMotorDatabase
from app.core.database import get_database
# Import schemas when needed to avoid circular imports
from app.services.timezone_service import get_local_now
import asyncio

logger = logging.getLogger(__name__)

class SettingsService:
    """
    Service to manage system settings with MongoDB storage and real-time synchronization
    """
    
    def __init__(self):
        self.db: Optional[AsyncIOMotorDatabase] = None
        self.current_settings: Optional[Dict[str, Any]] = None
        self.last_sync_time: Optional[datetime] = None
        self.settings_lock = asyncio.Lock()
        
    async def initialize(self):
        """Initialize the settings service"""
        try:
            self.db = await get_database()
            await self.load_settings_from_db()
            logger.info("✅ SettingsService initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize SettingsService: {e}")
            raise
    
    async def load_settings_from_db(self) -> Dict[str, Any]:
        """Load settings from MongoDB"""
        try:
            # Get the latest settings document
            settings_doc = await self.db.settings.find_one(
                {},
                sort=[("updated_at", -1)]
            )
            
            if settings_doc:
                # Remove MongoDB ObjectId for JSON serialization
                if "_id" in settings_doc:
                    del settings_doc["_id"]
                
                logger.info(f"🔧 Debug - Raw settings doc from MongoDB: {settings_doc}")
                
                self.current_settings = settings_doc
                self.last_sync_time = get_local_now()
                logger.info("📥 Settings loaded from database")
                return settings_doc
            else:
                # Create default settings if none exist
                default_settings = self._get_default_settings()
                await self.save_settings_to_db(default_settings)
                self.current_settings = default_settings
                self.last_sync_time = get_local_now()
                logger.info("🆕 Created default settings in database")
                return default_settings
                
        except Exception as e:
            logger.error(f"❌ Error loading settings from database: {e}")
            # Return default settings as fallback
            default_settings = self._get_default_settings()
            self.current_settings = default_settings
            return default_settings
    
    async def save_settings_to_db(self, settings: Dict[str, Any]) -> bool:
        """Save settings to MongoDB"""
        try:
            async with self.settings_lock:
                # Add timestamps
                settings["updated_at"] = get_local_now()
                if "created_at" not in settings:
                    settings["created_at"] = get_local_now()
                
                # Upsert settings document
                result = await self.db.settings.replace_one(
                    {},  # Match any document (we only keep one settings doc)
                    settings,
                    upsert=True
                )
                
                if result.acknowledged:
                    self.current_settings = settings
                    self.last_sync_time = get_local_now()
                    logger.info("💾 Settings saved to database successfully")
                    return True
                else:
                    logger.error("❌ Failed to save settings to database")
                    return False
                    
        except Exception as e:
            logger.error(f"❌ Error saving settings to database: {e}")
            return False
    
    async def get_current_settings(self) -> Dict[str, Any]:
        """Get current settings (from memory or database)"""
        if self.current_settings is None:
            await self.load_settings_from_db()
        
        return self.current_settings or self._get_default_settings()
    
    async def update_settings(self, settings_update: Dict[str, Any]) -> Dict[str, Any]:
        """Update settings and save to database"""
        try:
            current = await self.get_current_settings()
            
            # Deep merge the updates
            updated_settings = self._deep_merge(current, settings_update)
            
            # Save to database
            success = await self.save_settings_to_db(updated_settings)
            
            if success:
                logger.info(f"✅ Settings updated successfully")
                return updated_settings
            else:
                logger.error("❌ Failed to update settings")
                return current
                
        except Exception as e:
            logger.error(f"❌ Error updating settings: {e}")
            return await self.get_current_settings()
    
    async def sync_settings_on_startup(self) -> Dict[str, Any]:
        """Sync settings on server startup"""
        try:
            logger.info("🔄 Syncing settings on server startup...")
            settings = await self.load_settings_from_db()
            
            # Notify all services about the settings update
            await self._notify_services_of_settings_update(settings)
            
            logger.info("✅ Settings synced on startup")
            return settings
            
        except Exception as e:
            logger.error(f"❌ Error syncing settings on startup: {e}")
            return self._get_default_settings()
    
    async def _notify_services_of_settings_update(self, settings: Dict[str, Any]):
        """Notify all services that settings have been updated"""
        try:
            # Import here to avoid circular imports
            try:
                from app.services.email_service import update_email_settings
                
                # Update email service with new settings
                if "processing" in settings:
                    email_config = {
                        "notifications_enabled": settings["processing"].get("email_notifications_enabled", False),
                        "notification_email": settings["processing"].get("notification_email", ""),
                        "email_on_exit": settings["processing"].get("email_on_exit", True),
                        "email_on_entrance": settings["processing"].get("email_on_entrance", True)
                    }
                    logger.info(f"🔧 Debug - Processing settings from DB: {settings['processing']}")
                    logger.info(f"🔧 Debug - Email config created: {email_config}")
                    await update_email_settings(email_config)
                    logger.info("📧 Email service updated with new settings")
                    
            except ImportError as e:
                logger.warning(f"⚠️ Could not import email service for settings update: {e}")
            
            logger.info("🔔 All services notified of settings update")
            
        except Exception as e:
            logger.error(f"❌ Error notifying services of settings update: {e}")
    
    def _deep_merge(self, base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries"""
        result = base.copy()
        
        for key, value in update.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _get_default_settings(self) -> Dict[str, Any]:
        """Get default settings configuration"""
        return {
            "processing": {
                "detection_enabled": True,
                "classification_enabled": True,
                "computer_vision_fallback": True,
                "detection_confidence_threshold": 0.7,
                "classification_confidence_threshold": 0.7,
                "draw_bee_path": True,
                "draw_center_line": True,
                "draw_roi_box": True,
                "draw_status_text": True,
                "draw_confidence_scores": True,
                "draw_timestamp": True,
                "draw_frame_counter": True,
                "roi_x_min": 400,
                "roi_y_min": 0,
                "roi_x_max": 600,
                "roi_y_max": 700,
                "auto_delete_videos_after_session": False,
                "keep_original_videos": True,
                "auto_convert_videos": True,
                "video_buffer_seconds": 10,
                "email_notifications_enabled": False,
                "email_on_exit": True,
                "email_on_entrance": True,
                "notification_email": ""
            },
            "camera_config": {
                "internal_camera_id": "0",
                "external_camera_id": "1"
            },
            "created_at": get_local_now(),
            "updated_at": get_local_now()
        }

# Global settings service instance
settings_service = SettingsService()

async def get_settings_service() -> SettingsService:
    """Get the global settings service instance"""
    if settings_service.db is None:
        await settings_service.initialize()
    return settings_service