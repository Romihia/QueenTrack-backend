#!/usr/bin/env python3
"""
Test script to check for import errors before starting the server
"""
import sys
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_imports():
    """Test all critical imports"""
    try:
        logger.info("🔄 Testing imports...")
        
        # Test timezone service
        logger.info("Testing timezone service...")
        from app.services.timezone_service import get_timezone_service, get_local_now
        timezone_service = get_timezone_service()
        current_time = get_local_now()
        logger.info(f"✅ Timezone service OK - Current local time: {current_time}")
        
        # Test database
        logger.info("Testing database imports...")
        from app.core.database import get_database, init_db
        logger.info("✅ Database imports OK")
        
        # Test schemas
        logger.info("Testing schemas...")
        from app.schemas.schema import EventCreate, SettingsResponse, SettingsUpdate
        logger.info("✅ Schemas OK")
        
        # Test settings service
        logger.info("Testing settings service...")
        from app.services.settings_service import SettingsService, get_settings_service
        logger.info("✅ Settings service imports OK")
        
        # Test email service
        logger.info("Testing email service...")
        try:
            from app.services.email_service import EmailService, get_email_service, update_email_settings
            logger.info("✅ Email service imports OK")
        except ValueError as e:
            if "Email authentication incomplete" in str(e):
                logger.warning("⚠️ Email service requires EMAIL_USER and EMAIL_PASS env vars")
            else:
                raise
        
        # Test routes
        logger.info("Testing routes...")
        from app.routes.system_routes import router as system_router
        from app.routes.video_routes import router as video_router
        from app.routes.events_routes import router as events_router
        logger.info("✅ Routes OK")
        
        # Test main app
        logger.info("Testing main app...")
        from app.main import app
        logger.info("✅ Main app OK")
        
        logger.info("🎉 All imports successful!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Import error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_imports()
    if not success:
        sys.exit(1)
    
    print("\n🚀 All imports successful! Server should start without import errors.")