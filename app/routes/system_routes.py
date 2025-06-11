from fastapi import APIRouter, HTTPException
from app.services.email_service import email_service
from app.services.video_service import video_service
import logging
from datetime import datetime
import cv2
import os

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/health")
async def system_health_check():
    """
    Comprehensive system health check
    """
    try:
        health_status = {
            "timestamp": datetime.now().isoformat(),
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
    Test the complete system workflow
    """
    try:
        test_results = {
            "timestamp": datetime.now().isoformat(),
            "tests": {}
        }
        
        # Test 1: Email functionality
        try:
            email_test = email_service.send_bee_detection_notification(
                event_type="exit",
                timestamp=datetime.now(),
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
        
        # Test 3: Camera access (external camera test)
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