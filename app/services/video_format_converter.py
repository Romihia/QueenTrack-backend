"""
Video Format Converter Service - המרת פורמט וידאו לתאימות דפדפן
"""
import os
import subprocess
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import time

logger = logging.getLogger(__name__)

class VideoFormatConverter:
    """שירות להמרת פורמט וידאו מ-mp4v ל-avc1 לתאימות דפדפן"""
    
    def __init__(self, base_videos_dir: str = "/data/videos"):
        self.base_videos_dir = Path(base_videos_dir)
        self.converted_dir = self.base_videos_dir / "converted"
        self.converted_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if ffmpeg is available
        self.ffmpeg_available = self._check_ffmpeg_availability()
        if not self.ffmpeg_available:
            logger.error("FFmpeg not available - video conversion will not work")
        else:
            logger.info("✅ Video format converter initialized with FFmpeg support")
    
    def _check_ffmpeg_availability(self) -> bool:
        """בדוק אם ffmpeg זמין במערכת"""
        try:
            result = subprocess.run(['ffmpeg', '-version'], 
                                  capture_output=True, text=True, timeout=10)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
            logger.error(f"FFmpeg availability check failed: {e}")
            return False
    
    def convert_to_avc1(self, input_path: str, output_filename: Optional[str] = None) -> Optional[str]:
        """
        המר וידאו מ-mp4v ל-avc1 format
        
        Args:
            input_path: נתיב לקובץ הקלט
            output_filename: שם קובץ הפלט (אופציונלי)
            
        Returns:
            str: נתיב לקובץ המומר או None אם נכשל
        """
        if not self.ffmpeg_available:
            logger.error("Cannot convert video - FFmpeg not available")
            return None
        
        try:
            input_file = Path(input_path)
            
            if not input_file.exists():
                logger.error(f"Input video file not found: {input_path}")
                return None
            
            # יצירת שם קובץ פלט
            if output_filename is None:
                base_name = input_file.stem
                timestamp = int(time.time())
                output_filename = f"{base_name}_avc1_{timestamp}.mp4"
            
            output_path = self.converted_dir / output_filename
            output_path_str = str(output_path)
            
            logger.info(f"🔄 Starting video conversion: {input_path} -> {output_path_str}")
            
            # פקודת ffmpeg להמרה ל-avc1
            ffmpeg_cmd = [
                'ffmpeg',
                '-i', str(input_path),
                '-c:v', 'libx264',           # H.264 codec (avc1)
                '-profile:v', 'baseline',     # Baseline profile for maximum compatibility
                '-level', '3.0',              # Level 3.0 for wide compatibility
                '-c:a', 'aac',                # AAC audio codec
                '-b:a', '128k',               # Audio bitrate
                '-movflags', '+faststart',    # Enable fast start for web playback
                '-y',                         # Overwrite output file if exists
                output_path_str
            ]
            
            # הרצת הפקודה
            start_time = time.time()
            result = subprocess.run(
                ffmpeg_cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minutes timeout
            )
            
            conversion_time = time.time() - start_time
            
            if result.returncode == 0:
                # בדיקה שהקובץ נוצר ואינו ריק
                if output_path.exists() and output_path.stat().st_size > 0:
                    logger.info(f"✅ Video conversion successful in {conversion_time:.2f}s")
                    logger.info(f"   Input: {input_path} ({input_file.stat().st_size} bytes)")
                    logger.info(f"   Output: {output_path_str} ({output_path.stat().st_size} bytes)")
                    
                    return output_path_str
                else:
                    logger.error("Conversion appeared successful but output file is missing or empty")
                    return None
            else:
                logger.error(f"FFmpeg conversion failed with code {result.returncode}")
                logger.error(f"STDOUT: {result.stdout}")
                logger.error(f"STDERR: {result.stderr}")
                return None
                
        except subprocess.TimeoutExpired:
            logger.error(f"Video conversion timed out for: {input_path}")
            return None
        except Exception as e:
            logger.error(f"Video conversion failed for {input_path}: {e}")
            return None
    
    def convert_event_videos(self, internal_video_path: Optional[str], 
                           external_video_path: Optional[str]) -> Dict[str, Optional[str]]:
        """
        המר וידאו לאירוע (פנימי וחיצוני)
        
        Args:
            internal_video_path: נתיב וידאו פנימי
            external_video_path: נתיב וידאו חיצוני
            
        Returns:
            Dict: נתיבי הוידאו המומרים
        """
        results = {
            "internal_converted": None,
            "external_converted": None,
            "conversion_success": False,
            "errors": []
        }
        
        # המרת וידאו פנימי
        if internal_video_path and os.path.exists(internal_video_path):
            logger.info(f"Converting internal video: {internal_video_path}")
            converted_internal = self.convert_to_avc1(internal_video_path)
            if converted_internal:
                results["internal_converted"] = converted_internal
                logger.info(f"✅ Internal video converted successfully")
            else:
                error_msg = f"Failed to convert internal video: {internal_video_path}"
                logger.error(error_msg)
                results["errors"].append(error_msg)
        elif internal_video_path:
            error_msg = f"Internal video file not found: {internal_video_path}"
            logger.error(error_msg)
            results["errors"].append(error_msg)
        
        # המרת וידאו חיצוני
        if external_video_path and os.path.exists(external_video_path):
            logger.info(f"Converting external video: {external_video_path}")
            converted_external = self.convert_to_avc1(external_video_path)
            if converted_external:
                results["external_converted"] = converted_external
                logger.info(f"✅ External video converted successfully")
            else:
                error_msg = f"Failed to convert external video: {external_video_path}"
                logger.error(error_msg)
                results["errors"].append(error_msg)
        elif external_video_path:
            error_msg = f"External video file not found: {external_video_path}"
            logger.error(error_msg)
            results["errors"].append(error_msg)
        
        # בדיקה אם לפחות וידאו אחד הומר בהצלחה
        results["conversion_success"] = bool(
            results["internal_converted"] or results["external_converted"]
        )
        
        if results["conversion_success"]:
            logger.info("🎬 Event video conversion completed successfully")
        else:
            logger.error("❌ Event video conversion failed - no videos were converted")
        
        return results
    
    def get_video_url_from_path(self, full_path: str) -> Optional[str]:
        """
        המר נתיב מלא ל-URL יחסי לשירות הוידאו
        
        Args:
            full_path: הנתיב המלא לקובץ
            
        Returns:
            str: URL יחסי או None
        """
        try:
            path = Path(full_path)
            relative_path = path.relative_to(self.base_videos_dir)
            return f"/videos/{relative_path}"
        except ValueError:
            logger.error(f"Path {full_path} is not within videos directory")
            return None
        except Exception as e:
            logger.error(f"Error converting path to URL: {e}")
            return None
    
    def cleanup_original_videos(self, internal_path: Optional[str], 
                              external_path: Optional[str], 
                              keep_originals: bool = True) -> Dict[str, bool]:
        """
        נקה קבצי וידאו מקוריים לאחר המרה מוצלחת
        
        Args:
            internal_path: נתיב וידאו פנימי מקורי
            external_path: נתיב וידאו חיצוני מקורי
            keep_originals: האם לשמור את הקבצים המקוריים
            
        Returns:
            Dict: סטטוס הניקוי
        """
        results = {"internal_cleaned": False, "external_cleaned": False}
        
        if keep_originals:
            logger.info("Keeping original videos as requested")
            return results
        
        # מחיקת קובץ פנימי
        if internal_path and os.path.exists(internal_path):
            try:
                os.remove(internal_path)
                results["internal_cleaned"] = True
                logger.info(f"🗑️ Removed original internal video: {internal_path}")
            except Exception as e:
                logger.error(f"Failed to remove original internal video {internal_path}: {e}")
        
        # מחיקת קובץ חיצוני
        if external_path and os.path.exists(external_path):
            try:
                os.remove(external_path)
                results["external_cleaned"] = True
                logger.info(f"🗑️ Removed original external video: {external_path}")
            except Exception as e:
                logger.error(f"Failed to remove original external video {external_path}: {e}")
        
        return results
    
    def get_conversion_status(self) -> Dict[str, Any]:
        """קבל סטטוס שירות ההמרה"""
        converted_files = list(self.converted_dir.glob("*.mp4"))
        
        return {
            "ffmpeg_available": self.ffmpeg_available,
            "converted_directory": str(self.converted_dir),
            "converted_videos_count": len(converted_files),
            "converted_videos": [
                {
                    "filename": f.name,
                    "size_mb": round(f.stat().st_size / (1024 * 1024), 2),
                    "created": f.stat().st_ctime
                }
                for f in converted_files[:10]  # Show only last 10
            ]
        }

# Create singleton instance
video_format_converter = VideoFormatConverter() 