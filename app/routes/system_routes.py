from fastapi import APIRouter, HTTPException, Depends
from app.services.email_service import email_service
from app.services.video_service import video_service
from app.services.video_format_converter import video_format_converter
from app.services.settings_service import get_settings_service, SettingsService
# Schemas imported when needed to avoid circular imports
import logging
from datetime import datetime
import cv2
import os
from typing import Dict, Any
from app.services.timezone_service import get_local_now

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/health")
async def system_health_check():
    """
    Comprehensive system health check
    """
    try:
        health_status = {
            "timestamp": get_local_now().isoformat(),
            "overall_status": "healthy",
            "components": {}
        }
        
        # Check email service
        try:
            email_test = email_service.test_email_connection()
            health_status["components"]["email_service"] = {
                "status": "healthy" if email_test else "unhealthy",
                "connection_test": email_test,
                "configured": bool(email_service.email_user and email_service.email_pass)
            }
        except Exception as e:
            health_status["components"]["email_service"] = {
                "status": "error",
                "error": str(e)
            }
        
        # Check video service
        try:
            videos = video_service.list_videos()
            health_status["components"]["video_service"] = {
                "status": "healthy",
                "videos_count": sum(len(v) for v in videos.values()),
                "directories": list(videos.keys())
            }
        except Exception as e:
            health_status["components"]["video_service"] = {
                "status": "error",
                "error": str(e)
            }
        
        # Check video conversion service
        try:
            conversion_status = video_format_converter.get_conversion_status()
            health_status["components"]["video_conversion"] = {
                "status": "healthy" if conversion_status["ffmpeg_available"] else "warning",
                "ffmpeg_available": conversion_status["ffmpeg_available"],
                "converted_videos_count": conversion_status["converted_videos_count"],
                "conversion_directory": conversion_status["converted_directory"]
            }
        except Exception as e:
            health_status["components"]["video_conversion"] = {
                "status": "error",
                "error": str(e)
            }
        
        # Check camera availability
        available_cameras = []
        try:
            for i in range(5):  # Check first 5 camera indices
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    available_cameras.append({
                        "id": i,
                        "width": width,
                        "height": height,
                        "fps": fps
                    })
                cap.release()
            
            health_status["components"]["cameras"] = {
                "status": "healthy" if available_cameras else "warning",
                "available_cameras": available_cameras,
                "count": len(available_cameras)
            }
        except Exception as e:
            health_status["components"]["cameras"] = {
                "status": "error",
                "error": str(e)
            }
        
        # Check YOLO models
        try:
            model_files = [
                ("yolov8n.pt", "Detection model"),
                ("best.pt", "Classification model")
            ]
            
            models_status = {}
            for model_file, description in model_files:
                if os.path.exists(model_file):
                    models_status[model_file] = {
                        "status": "available",
                        "description": description,
                        "size_mb": round(os.path.getsize(model_file) / (1024*1024), 2)
                    }
                else:
                    models_status[model_file] = {
                        "status": "missing",
                        "description": description
                    }
            
            health_status["components"]["yolo_models"] = {
                "status": "healthy" if all(m["status"] == "available" for m in models_status.values()) else "warning",
                "models": models_status
            }
        except Exception as e:
            health_status["components"]["yolo_models"] = {
                "status": "error",
                "error": str(e)
            }
        
        # Determine overall status
        component_statuses = [comp.get("status", "error") for comp in health_status["components"].values()]
        if "error" in component_statuses:
            health_status["overall_status"] = "unhealthy"
        elif "warning" in component_statuses:
            health_status["overall_status"] = "warning"
        else:
            health_status["overall_status"] = "healthy"
        
        return health_status
        
    except Exception as e:
        logger.error(f"System health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@router.post("/test-full-system")
async def test_full_system():
    """
    Test the complete system workflow including video conversion
    """
    try:
        test_results = {
            "timestamp": get_local_now().isoformat(),
            "tests": {}
        }
        
        # Test 1: Email functionality
        try:
            email_test = email_service.send_bee_detection_notification(
                event_type="exit",
                timestamp=get_local_now(),
                additional_info={"test": True, "full_system_test": True}
            )
            test_results["tests"]["email_notification"] = {
                "status": "passed" if email_test else "failed",
                "message": "Email notification test completed"
            }
        except Exception as e:
            test_results["tests"]["email_notification"] = {
                "status": "error",
                "error": str(e)
            }
        
        # Test 2: Video service
        try:
            videos = video_service.list_videos()
            test_results["tests"]["video_listing"] = {
                "status": "passed",
                "videos_found": sum(len(v) for v in videos.values())
            }
        except Exception as e:
            test_results["tests"]["video_listing"] = {
                "status": "error",
                "error": str(e)
            }
        
        # Test 3: Video conversion service
        try:
            conversion_status = video_format_converter.get_conversion_status()
            test_results["tests"]["video_conversion"] = {
                "status": "passed" if conversion_status["ffmpeg_available"] else "warning",
                "ffmpeg_available": conversion_status["ffmpeg_available"],
                "message": "FFmpeg available and ready" if conversion_status["ffmpeg_available"] else "FFmpeg not available"
            }
        except Exception as e:
            test_results["tests"]["video_conversion"] = {
                "status": "error",
                "error": str(e)
            }
        
        # Test 4: Camera access (external camera test)
        try:
            camera_test_passed = False
            for i in range(3):  # Test first 3 cameras
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        camera_test_passed = True
                        test_results["tests"]["camera_access"] = {
                            "status": "passed",
                            "working_camera_id": i,
                            "frame_shape": frame.shape
                        }
                        cap.release()
                        break
                cap.release()
            
            if not camera_test_passed:
                test_results["tests"]["camera_access"] = {
                    "status": "warning",
                    "message": "No working cameras found"
                }
                
        except Exception as e:
            test_results["tests"]["camera_access"] = {
                "status": "error",
                "error": str(e)
            }
        
        # Determine overall test result
        test_statuses = [test.get("status", "error") for test in test_results["tests"].values()]
        if "error" in test_statuses:
            test_results["overall_result"] = "failed"
        elif "failed" in test_statuses:
            test_results["overall_result"] = "failed"
        elif "warning" in test_statuses:
            test_results["overall_result"] = "warning"
        else:
            test_results["overall_result"] = "passed"
        
        return test_results
        
    except Exception as e:
        logger.error(f"Full system test failed: {e}")
        raise HTTPException(status_code=500, detail=f"System test failed: {str(e)}")

# Settings Management Endpoints

@router.get("/settings")
async def get_current_settings(
    settings_service: SettingsService = Depends(get_settings_service)
) -> Dict[str, Any]:
    """Get current system settings"""
    try:
        settings = await settings_service.get_current_settings()
        logger.info("📋 Settings retrieved successfully")
        return {
            "status": "success",
            "settings": settings
        }
    except Exception as e:
        logger.error(f"❌ Error retrieving settings: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve settings: {str(e)}")

@router.post("/settings")
async def update_system_settings(
    settings_data: Dict[str, Any],
    settings_service: SettingsService = Depends(get_settings_service)
) -> Dict[str, Any]:
    """Update system settings and sync with all services"""
    try:
        # Update settings in database
        updated_settings = await settings_service.update_settings(settings_data)
        
        # Notify all services of the update
        await settings_service._notify_services_of_settings_update(updated_settings)
        
        logger.info("✅ Settings updated and synced successfully")
        return {
            "status": "success",
            "message": "Settings updated successfully",
            "settings": updated_settings
        }
    except Exception as e:
        logger.error(f"❌ Error updating settings: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update settings: {str(e)}")

@router.post("/settings/sync")
async def sync_settings_on_startup(
    settings_service: SettingsService = Depends(get_settings_service)
) -> Dict[str, Any]:
    """Sync settings on server startup - loads from database and updates all services"""
    try:
        settings = await settings_service.sync_settings_on_startup()
        
        logger.info("🔄 Settings synced on startup")
        return {
            "status": "success",
            "message": "Settings synced successfully",
            "settings": settings,
            "sync_time": get_local_now().isoformat()
        }
    except Exception as e:
        logger.error(f"❌ Error syncing settings on startup: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to sync settings: {str(e)}")

@router.get("/settings/presets")
async def get_settings_presets() -> Dict[str, Any]:
    """Get available settings presets"""
    try:
        presets = {
            "high_accuracy": {
                "name": "High Accuracy",
                "description": "Maximum detection accuracy with all features enabled",
                "processing": {
                    "detection_enabled": True,
                    "classification_enabled": True,
                    "computer_vision_fallback": True,
                    "detection_confidence_threshold": 0.8,
                    "classification_confidence_threshold": 0.8,
                    "draw_bee_path": True,
                    "draw_center_line": True,
                    "draw_roi_box": True,
                    "draw_status_text": True,
                    "draw_confidence_scores": True,
                    "draw_timestamp": True,
                    "draw_frame_counter": True
                }
            },
            "performance": {
                "name": "Performance Mode",
                "description": "Optimized for speed with reduced visual elements",
                "processing": {
                    "detection_enabled": True,
                    "classification_enabled": True,
                    "computer_vision_fallback": False,
                    "detection_confidence_threshold": 0.6,
                    "classification_confidence_threshold": 0.6,
                    "draw_bee_path": False,
                    "draw_center_line": True,
                    "draw_roi_box": False,
                    "draw_status_text": False,
                    "draw_confidence_scores": False,
                    "draw_timestamp": False,
                    "draw_frame_counter": False
                }
            },
            "minimal": {
                "name": "Minimal Detection",
                "description": "Basic detection with minimal processing",
                "processing": {
                    "detection_enabled": True,
                    "classification_enabled": False,
                    "computer_vision_fallback": False,
                    "detection_confidence_threshold": 0.5,
                    "classification_confidence_threshold": 0.5,
                    "draw_bee_path": False,
                    "draw_center_line": True,
                    "draw_roi_box": False,
                    "draw_status_text": False,
                    "draw_confidence_scores": False,
                    "draw_timestamp": False,
                    "draw_frame_counter": False
                }
            }
        }
        
        return {
            "status": "success",
            "presets": presets
        }
    except Exception as e:
        logger.error(f"❌ Error retrieving presets: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve presets: {str(e)}")

@router.post("/settings/apply-preset/{preset_name}")
async def apply_settings_preset(
    preset_name: str,
    settings_service: SettingsService = Depends(get_settings_service)
) -> Dict[str, Any]:
    """Apply a settings preset"""
    try:
        # Get presets
        presets_response = await get_settings_presets()
        presets = presets_response["presets"]
        
        if preset_name not in presets:
            raise HTTPException(status_code=404, detail=f"Preset '{preset_name}' not found")
        
        # Get current settings and merge with preset
        current_settings = await settings_service.get_current_settings()
        preset_data = presets[preset_name]
        
        # Update only the processing section from the preset
        updated_settings = await settings_service.update_settings({
            "processing": {
                **current_settings.get("processing", {}),
                **preset_data["processing"]
            }
        })
        
        logger.info(f"✅ Applied preset '{preset_name}' successfully")
        return {
            "status": "success",
            "message": f"Applied {preset_data['name']} preset successfully",
            "settings": updated_settings
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error applying preset '{preset_name}': {e}")
        raise HTTPException(status_code=500, detail=f"Failed to apply preset: {str(e)}")

@router.post("/settings/reset")
async def reset_settings_to_defaults(
    settings_service: SettingsService = Depends(get_settings_service)
) -> Dict[str, Any]:
    """Reset all settings to default values"""
    try:
        # Get default settings and save them
        default_settings = settings_service._get_default_settings()
        updated_settings = await settings_service.update_settings(default_settings)
        
        logger.info("🔄 Settings reset to defaults successfully")
        return {
            "status": "success",
            "message": "Settings reset to defaults successfully",
            "settings": updated_settings
        }
    except Exception as e:
        logger.error(f"❌ Error resetting settings: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to reset settings: {str(e)}")

@router.post("/settings/reload-email")
async def reload_email_settings() -> Dict[str, Any]:
    """Manually reload email settings from database"""
    try:
        from app.services.email_service import get_email_service
        
        email_service = get_email_service()
        await email_service.load_settings_from_database()
        
        return {
            "status": "success",
            "message": "Email settings reloaded from database",
            "email_settings": {
                "notifications_enabled": email_service.notifications_enabled,
                "notification_email": email_service.notification_email,
                "email_on_exit": email_service.email_on_exit,
                "email_on_entrance": email_service.email_on_entrance,
                "recipient_email": email_service.get_recipient_email()
            }
        }
    except Exception as e:
        logger.error(f"❌ Error reloading email settings: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to reload email settings: {str(e)}")


@router.post("/test-email")
async def test_email_notification() -> Dict[str, Any]:
    """Test email notification with current settings"""
    try:
        from app.services.email_service import get_email_service
        
        email_service = get_email_service()
        
        # Reload settings first
        await email_service.load_settings_from_database()
        
        # Test email functionality
        success = email_service.send_bee_detection_notification(
            event_type="exit",
            timestamp=get_local_now(),
            additional_info={"test": True, "manual_test": True}
        )
        
        return {
            "status": "success" if success else "failed",
            "message": "Test email sent successfully" if success else "Test email failed to send",
            "email_settings": {
                "notifications_enabled": email_service.notifications_enabled,
                "notification_email": email_service.notification_email,
                "recipient_email": email_service.get_recipient_email(),
                "is_notifications_enabled": email_service.is_notifications_enabled()
            }
        }
    except Exception as e:
        logger.error(f"❌ Error testing email: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to test email: {str(e)}") 