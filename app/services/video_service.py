import os
import logging
import mimetypes
from pathlib import Path
from typing import Optional, Tuple
from fastapi import HTTPException
from fastapi.responses import StreamingResponse, FileResponse
import cv2

# Ensure logs directory exists
os.makedirs('/data/logs', exist_ok=True)

# Configure logging
logger = logging.getLogger(__name__)

class VideoService:
    """
    Professional video service for Queen Track system
    Handles video storage, serving, and streaming
    """
    
    def __init__(self, base_videos_dir: str = "/data/videos"):
        self.base_videos_dir = Path(base_videos_dir)
        self.base_videos_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.subdirs = {
            'outside_videos': self.base_videos_dir / 'outside_videos',
            'temp_videos': self.base_videos_dir / 'temp_videos',
            'uploaded_videos': self.base_videos_dir / 'uploaded_videos',
            'processed_videos': self.base_videos_dir / 'processed_videos'
        }
        
        for subdir in self.subdirs.values():
            subdir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Video service initialized. Base directory: {self.base_videos_dir}")

    def get_video_path(self, relative_path: str) -> Path:
        """
        Get absolute path for a video file
        
        Args:
            relative_path: Relative path from videos directory
            
        Returns:
            Path: Absolute path to video file
        """
        # Remove leading slash if present
        relative_path = relative_path.lstrip('/')
        
        # Remove 'videos/' prefix if present (for URLs like /videos/outside_videos/video.mp4)
        if relative_path.startswith('videos/'):
            relative_path = relative_path[7:]
        
        full_path = self.base_videos_dir / relative_path
        
        # Security check - ensure path is within base directory
        try:
            full_path.resolve().relative_to(self.base_videos_dir.resolve())
        except ValueError:
            logger.error(f"Security violation: Path {relative_path} is outside base directory")
            raise HTTPException(status_code=403, detail="Access forbidden")
        
        return full_path

    def file_exists(self, relative_path: str) -> bool:
        """
        Check if video file exists
        
        Args:
            relative_path: Relative path from videos directory
            
        Returns:
            bool: True if file exists, False otherwise
        """
        try:
            path = self.get_video_path(relative_path)
            exists = path.exists() and path.is_file()
            logger.info(f"File existence check for {relative_path}: {exists}")
            return exists
        except Exception as e:
            logger.error(f"Error checking file existence for {relative_path}: {e}")
            return False

    def get_video_info(self, relative_path: str) -> Optional[dict]:
        """
        Get video file information
        
        Args:
            relative_path: Relative path from videos directory
            
        Returns:
            dict: Video information or None if file doesn't exist
        """
        try:
            path = self.get_video_path(relative_path)
            
            if not path.exists():
                return None
            
            # Get file stats
            stat = path.stat()
            
            # Get video properties using OpenCV
            cap = cv2.VideoCapture(str(path))
            
            info = {
                'filename': path.name,
                'size_bytes': stat.st_size,
                'size_mb': round(stat.st_size / (1024 * 1024), 2),
                'modified_time': stat.st_mtime,
                'duration_seconds': None,
                'fps': None,
                'frame_count': None,
                'width': None,
                'height': None
            }
            
            if cap.isOpened():
                info['fps'] = cap.get(cv2.CAP_PROP_FPS)
                info['frame_count'] = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                info['width'] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                info['height'] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                if info['fps'] > 0 and info['frame_count'] > 0:
                    info['duration_seconds'] = round(info['frame_count'] / info['fps'], 2)
                
                cap.release()
            
            logger.info(f"Video info retrieved for {relative_path}: {info}")
            return info
            
        except Exception as e:
            logger.error(f"Error getting video info for {relative_path}: {e}")
            return None

    def get_file_response(self, relative_path: str) -> FileResponse:
        """
        Get FileResponse for video streaming
        
        Args:
            relative_path: Relative path from videos directory
            
        Returns:
            FileResponse: FastAPI file response for streaming
        """
        path = self.get_video_path(relative_path)
        
        if not path.exists():
            logger.error(f"Video file not found: {relative_path}")
            raise HTTPException(status_code=404, detail="Video file not found")
        
        # Determine MIME type
        mime_type, _ = mimetypes.guess_type(str(path))
        if mime_type is None:
            mime_type = "video/mp4"  # Default for video files
        
        logger.info(f"Serving video file: {relative_path} (MIME: {mime_type})")
        
        return FileResponse(
            path=str(path),
            media_type=mime_type,
            filename=path.name,
            headers={
                "Accept-Ranges": "bytes",
                "Cache-Control": "public, max-age=3600",
                "Content-Disposition": f"inline; filename={path.name}"
            }
        )

    def get_streaming_response(self, relative_path: str, range_header: Optional[str] = None) -> StreamingResponse:
        """
        Get streaming response with range support for video playback
        
        Args:
            relative_path: Relative path from videos directory
            range_header: HTTP Range header value
            
        Returns:
            StreamingResponse: Streaming response with proper headers
        """
        path = self.get_video_path(relative_path)
        
        if not path.exists():
            logger.error(f"Video file not found: {relative_path}")
            raise HTTPException(status_code=404, detail="Video file not found")
        
        file_size = path.stat().st_size
        
        # Determine MIME type
        mime_type, _ = mimetypes.guess_type(str(path))
        if mime_type is None:
            mime_type = "video/mp4"
        
        # Handle range requests for video streaming
        start = 0
        end = file_size - 1
        
        if range_header:
            try:
                # Parse range header (e.g., "bytes=0-1023")
                range_match = range_header.replace('bytes=', '').split('-')
                start = int(range_match[0]) if range_match[0] else 0
                end = int(range_match[1]) if range_match[1] else file_size - 1
            except Exception as e:
                logger.warning(f"Invalid range header: {range_header}, error: {e}")
        
        # Ensure valid range
        start = max(0, start)
        end = min(file_size - 1, end)
        content_length = end - start + 1
        
        def generate_chunks():
            with open(path, 'rb') as f:
                f.seek(start)
                remaining = content_length
                chunk_size = 8192  # 8KB chunks
                
                while remaining > 0:
                    chunk_size_to_read = min(chunk_size, remaining)
                    chunk = f.read(chunk_size_to_read)
                    if not chunk:
                        break
                    remaining -= len(chunk)
                    yield chunk
        
        headers = {
            "Accept-Ranges": "bytes",
            "Content-Length": str(content_length),
            "Content-Range": f"bytes {start}-{end}/{file_size}",
            "Cache-Control": "public, max-age=3600"
        }
        
        status_code = 206 if range_header else 200
        
        logger.info(f"Streaming video: {relative_path} (Range: {start}-{end}/{file_size})")
        
        return StreamingResponse(
            generate_chunks(),
            status_code=status_code,
            headers=headers,
            media_type=mime_type
        )

    def save_video_locally(self, file_path: str) -> Optional[str]:
        """
        Process video file and return URL for accessing it
        
        Args:
            file_path: Absolute path to video file
            
        Returns:
            str: Relative URL for accessing the video or None if failed
        """
        try:
            path = Path(file_path)
            
            if not path.exists():
                logger.error(f"Video file does not exist: {file_path}")
                return None
            
            # If file is already in videos directory, create relative path
            try:
                relative_path = path.relative_to(self.base_videos_dir)
                video_url = f"/videos/{relative_path}"
            except ValueError:
                # File is outside videos directory, copy it
                filename = path.name
                destination = self.subdirs['processed_videos'] / filename
                
                # Copy file if it doesn't exist in destination
                if not destination.exists():
                    import shutil
                    shutil.copy2(path, destination)
                    logger.info(f"Video copied to: {destination}")
                
                relative_path = destination.relative_to(self.base_videos_dir)
                video_url = f"/videos/{relative_path}"
            
            logger.info(f"Video saved locally: {file_path}")
            logger.info(f"Video accessible at: {video_url}")
            
            return video_url
            
        except Exception as e:
            logger.error(f"Failed to process local video {file_path}: {e}")
            return None

    def list_videos(self, subdir: Optional[str] = None) -> dict:
        """
        List all videos in directory or subdirectory
        
        Args:
            subdir: Specific subdirectory to list (optional)
            
        Returns:
            dict: Videos organized by directory
        """
        videos = {}
        
        try:
            if subdir and subdir in self.subdirs:
                # List specific subdirectory
                directory = self.subdirs[subdir]
                videos[subdir] = []
                
                for video_file in directory.glob("*.mp4"):
                    relative_path = video_file.relative_to(self.base_videos_dir)
                    video_info = self.get_video_info(str(relative_path))
                    if video_info:
                        video_info['url'] = f"/videos/{relative_path}"
                        videos[subdir].append(video_info)
            else:
                # List all subdirectories
                for dir_name, directory in self.subdirs.items():
                    videos[dir_name] = []
                    
                    for video_file in directory.glob("*.mp4"):
                        relative_path = video_file.relative_to(self.base_videos_dir)
                        video_info = self.get_video_info(str(relative_path))
                        if video_info:
                            video_info['url'] = f"/videos/{relative_path}"
                            videos[dir_name].append(video_info)
            
            logger.info(f"Listed videos: {sum(len(v) for v in videos.values())} total")
            return videos
            
        except Exception as e:
            logger.error(f"Error listing videos: {e}")
            return {}

# Create singleton instance
video_service = VideoService() 