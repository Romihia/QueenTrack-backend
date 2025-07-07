"""
Video Compression and Storage Optimization Service
Advanced video processing with intelligent compression and storage management
"""
import os
import subprocess
import asyncio
import logging
import time
import shutil
import json
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor
import hashlib
import sqlite3

logger = logging.getLogger(__name__)

@dataclass
class CompressionProfile:
    """Video compression profile configuration"""
    name: str
    description: str
    video_codec: str = "libx264"
    audio_codec: str = "aac"
    crf: int = 23  # Constant Rate Factor (lower = better quality)
    preset: str = "medium"  # Encoding speed preset
    resolution_scale: Optional[str] = None  # e.g., "1280:720" or None for original
    frame_rate: Optional[int] = None  # Target FPS or None for original
    max_bitrate: Optional[str] = None  # e.g., "2000k" 
    audio_bitrate: str = "128k"
    additional_params: List[str] = field(default_factory=list)

@dataclass
class VideoInfo:
    """Video file information"""
    file_path: str
    file_size_mb: float
    duration_seconds: float
    resolution: str
    frame_rate: float
    codec: str
    bitrate: Optional[int]
    creation_time: datetime
    checksum: str

@dataclass
class CompressionJob:
    """Video compression job"""
    job_id: str
    input_path: str
    output_path: str
    profile: CompressionProfile
    status: str = "pending"  # pending, processing, completed, failed
    progress: float = 0.0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error_message: Optional[str] = None
    original_size_mb: float = 0.0
    compressed_size_mb: float = 0.0
    compression_ratio: float = 0.0

class VideoCompressionService:
    """Advanced video compression and storage optimization service"""
    
    def __init__(self, base_storage_dir: str = "/data/videos"):
        self.base_storage_dir = Path(base_storage_dir)
        self.compressed_dir = self.base_storage_dir / "compressed"
        self.temp_dir = self.base_storage_dir / "temp_compression"
        self.archive_dir = self.base_storage_dir / "archive"
        
        # Create directories
        for directory in [self.compressed_dir, self.temp_dir, self.archive_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Database for tracking compression jobs and video metadata
        self.db_path = self.base_storage_dir / "compression_db.sqlite"
        self._init_database()
        
        # Configuration
        self.max_concurrent_jobs = 2
        self.auto_cleanup_enabled = True
        self.cleanup_older_than_days = 30
        self.max_storage_gb = 50
        self.compression_queue_size = 100
        
        # Threading
        self.executor = ThreadPoolExecutor(max_workers=self.max_concurrent_jobs)
        self.active_jobs: Dict[str, CompressionJob] = {}
        self.job_queue: List[CompressionJob] = []
        
        # Compression profiles
        self.profiles = self._create_default_profiles()
        
        # Background tasks
        self.cleanup_task = None
        self.monitor_task = None
        self.processing_active = False
        
        # Statistics
        self.stats = {
            "total_videos_processed": 0,
            "total_space_saved_gb": 0.0,
            "total_processing_time_hours": 0.0,
            "average_compression_ratio": 0.0,
            "failed_jobs": 0
        }
        
        logger.info("🎬 Video Compression Service initialized")
        
        # Start background tasks
        self._start_background_tasks()
    
    def _init_database(self):
        """Initialize SQLite database for tracking videos and jobs"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS videos (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        file_path TEXT UNIQUE NOT NULL,
                        file_size_mb REAL NOT NULL,
                        duration_seconds REAL NOT NULL,
                        resolution TEXT NOT NULL,
                        frame_rate REAL NOT NULL,
                        codec TEXT NOT NULL,
                        bitrate INTEGER,
                        creation_time TEXT NOT NULL,
                        checksum TEXT NOT NULL,
                        compressed_path TEXT,
                        compression_ratio REAL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS compression_jobs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        job_id TEXT UNIQUE NOT NULL,
                        input_path TEXT NOT NULL,
                        output_path TEXT NOT NULL,
                        profile_name TEXT NOT NULL,
                        status TEXT NOT NULL,
                        progress REAL DEFAULT 0.0,
                        start_time TEXT,
                        end_time TEXT,
                        error_message TEXT,
                        original_size_mb REAL DEFAULT 0.0,
                        compressed_size_mb REAL DEFAULT 0.0,
                        compression_ratio REAL DEFAULT 0.0,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                conn.commit()
                logger.info("📚 Compression database initialized")
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
    
    def _create_default_profiles(self) -> Dict[str, CompressionProfile]:
        """Create default compression profiles"""
        profiles = {
            "high_quality": CompressionProfile(
                name="high_quality",
                description="High quality with minimal compression",
                crf=18,
                preset="slow",
                max_bitrate="5000k"
            ),
            "balanced": CompressionProfile(
                name="balanced",
                description="Balanced quality and file size",
                crf=23,
                preset="medium",
                max_bitrate="2000k"
            ),
            "space_saver": CompressionProfile(
                name="space_saver",
                description="Maximum compression for storage",
                crf=28,
                preset="fast",
                resolution_scale="1280:720",
                frame_rate=24,
                max_bitrate="1000k"
            ),
            "archive": CompressionProfile(
                name="archive",
                description="Long-term archival with very high compression",
                crf=32,
                preset="veryslow",
                resolution_scale="960:540",
                frame_rate=15,
                max_bitrate="500k",
                audio_bitrate="64k"
            ),
            "web_optimized": CompressionProfile(
                name="web_optimized",
                description="Optimized for web streaming",
                crf=25,
                preset="fast",
                resolution_scale="1280:720",
                frame_rate=30,
                max_bitrate="1500k",
                additional_params=["-movflags", "+faststart"]
            )
        }
        return profiles
    
    def _start_background_tasks(self):
        """Start background processing tasks"""
        self.processing_active = True
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
        self.monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("🚀 Background compression tasks started")
    
    async def _cleanup_loop(self):
        """Background cleanup of old videos and temporary files"""
        while self.processing_active:
            try:
                await asyncio.sleep(3600)  # Run every hour
                
                if self.auto_cleanup_enabled:
                    await self._perform_cleanup()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
    
    async def _monitor_loop(self):
        """Monitor storage usage and performance"""
        while self.processing_active:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes
                
                # Check storage usage
                storage_usage = await self._get_storage_usage()
                if storage_usage["used_gb"] > self.max_storage_gb:
                    logger.warning(f"Storage usage ({storage_usage['used_gb']:.1f}GB) exceeds limit ({self.max_storage_gb}GB)")
                    await self._emergency_cleanup()
                
                # Process queued jobs
                await self._process_job_queue()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitor loop: {e}")
    
    async def _perform_cleanup(self):
        """Perform routine cleanup of old files"""
        try:
            cutoff_date = datetime.now() - timedelta(days=self.cleanup_older_than_days)
            cleanup_count = 0
            space_freed = 0.0
            
            # Clean up old temporary files
            for temp_file in self.temp_dir.glob("*"):
                if temp_file.stat().st_mtime < cutoff_date.timestamp():
                    size_mb = temp_file.stat().st_size / 1024 / 1024
                    temp_file.unlink()
                    cleanup_count += 1
                    space_freed += size_mb
            
            # Archive old videos
            await self._archive_old_videos(cutoff_date)
            
            # Clean up database entries for deleted files
            await self._cleanup_database()
            
            if cleanup_count > 0:
                logger.info(f"🧹 Cleanup completed: {cleanup_count} files removed, {space_freed:.1f}MB freed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    async def _emergency_cleanup(self):
        """Emergency cleanup when storage is full"""
        logger.warning("🚨 Emergency cleanup triggered - storage limit exceeded")
        
        try:
            # Move oldest videos to archive with maximum compression
            videos_to_archive = await self._get_oldest_videos(limit=10)
            
            for video_info in videos_to_archive:
                await self.compress_video_async(
                    video_info["file_path"],
                    profile_name="archive"
                )
            
            # Remove original files after archival compression
            await self._cleanup_archived_originals()
            
        except Exception as e:
            logger.error(f"Error during emergency cleanup: {e}")
    
    async def _process_job_queue(self):
        """Process queued compression jobs"""
        while self.job_queue and len(self.active_jobs) < self.max_concurrent_jobs:
            job = self.job_queue.pop(0)
            self.active_jobs[job.job_id] = job
            
            # Start job in thread pool
            loop = asyncio.get_event_loop()
            loop.run_in_executor(self.executor, self._process_compression_job, job)
    
    def _process_compression_job(self, job: CompressionJob):
        """Process a single compression job"""
        try:
            job.status = "processing"
            job.start_time = datetime.now()
            self._update_job_in_database(job)
            
            logger.info(f"🎬 Starting compression job {job.job_id}")
            
            # Get original file info
            job.original_size_mb = os.path.getsize(job.input_path) / 1024 / 1024
            
            # Build FFmpeg command
            ffmpeg_cmd = self._build_ffmpeg_command(job.input_path, job.output_path, job.profile)
            
            # Execute compression
            result = subprocess.run(
                ffmpeg_cmd,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )
            
            if result.returncode == 0:
                # Compression successful
                job.compressed_size_mb = os.path.getsize(job.output_path) / 1024 / 1024
                job.compression_ratio = job.original_size_mb / job.compressed_size_mb if job.compressed_size_mb > 0 else 0
                job.status = "completed"
                job.progress = 100.0
                
                # Update statistics
                self.stats["total_videos_processed"] += 1
                self.stats["total_space_saved_gb"] += (job.original_size_mb - job.compressed_size_mb) / 1024
                
                logger.info(f"✅ Compression completed: {job.job_id} (ratio: {job.compression_ratio:.2f}x)")
                
            else:
                # Compression failed
                job.status = "failed"
                job.error_message = result.stderr
                self.stats["failed_jobs"] += 1
                
                logger.error(f"❌ Compression failed: {job.job_id} - {result.stderr}")
            
        except subprocess.TimeoutExpired:
            job.status = "failed"
            job.error_message = "Compression timeout"
            logger.error(f"⏰ Compression timeout: {job.job_id}")
            
        except Exception as e:
            job.status = "failed"
            job.error_message = str(e)
            logger.error(f"💥 Compression error: {job.job_id} - {e}")
            
        finally:
            job.end_time = datetime.now()
            if job.start_time:
                duration_hours = (job.end_time - job.start_time).total_seconds() / 3600
                self.stats["total_processing_time_hours"] += duration_hours
            
            self._update_job_in_database(job)
            
            # Remove from active jobs
            if job.job_id in self.active_jobs:
                del self.active_jobs[job.job_id]
    
    def _build_ffmpeg_command(self, input_path: str, output_path: str, profile: CompressionProfile) -> List[str]:
        """Build FFmpeg command for compression"""
        cmd = ["ffmpeg", "-i", input_path]
        
        # Video encoding parameters
        cmd.extend(["-c:v", profile.video_codec])
        cmd.extend(["-crf", str(profile.crf)])
        cmd.extend(["-preset", profile.preset])
        
        # Resolution scaling
        if profile.resolution_scale:
            cmd.extend(["-vf", f"scale={profile.resolution_scale}"])
        
        # Frame rate
        if profile.frame_rate:
            cmd.extend(["-r", str(profile.frame_rate)])
        
        # Bitrate limit
        if profile.max_bitrate:
            cmd.extend(["-maxrate", profile.max_bitrate])
            cmd.extend(["-bufsize", f"{int(profile.max_bitrate.rstrip('k')) * 2}k"])
        
        # Audio encoding
        cmd.extend(["-c:a", profile.audio_codec])
        cmd.extend(["-b:a", profile.audio_bitrate])
        
        # Additional parameters
        cmd.extend(profile.additional_params)
        
        # Output options
        cmd.extend(["-y", output_path])  # -y to overwrite output file
        
        return cmd
    
    async def compress_video_async(self, input_path: str, profile_name: str = "balanced", 
                                 output_path: Optional[str] = None) -> str:
        """Queue video for compression asynchronously"""
        if profile_name not in self.profiles:
            raise ValueError(f"Unknown compression profile: {profile_name}")
        
        profile = self.profiles[profile_name]
        
        # Generate output path if not provided
        if output_path is None:
            input_file = Path(input_path)
            output_path = str(self.compressed_dir / f"{input_file.stem}_{profile_name}.mp4")
        
        # Create compression job
        job_id = f"job_{int(time.time() * 1000)}"
        job = CompressionJob(
            job_id=job_id,
            input_path=input_path,
            output_path=output_path,
            profile=profile
        )
        
        # Add to queue
        if len(self.job_queue) < self.compression_queue_size:
            self.job_queue.append(job)
            self._save_job_to_database(job)
            
            logger.info(f"📋 Compression job queued: {job_id} ({profile_name})")
            return job_id
        else:
            raise RuntimeError("Compression queue is full")
    
    def compress_video_sync(self, input_path: str, profile_name: str = "balanced", 
                          output_path: Optional[str] = None) -> Dict[str, Any]:
        """Compress video synchronously"""
        if profile_name not in self.profiles:
            raise ValueError(f"Unknown compression profile: {profile_name}")
        
        profile = self.profiles[profile_name]
        
        # Generate output path if not provided
        if output_path is None:
            input_file = Path(input_path)
            output_path = str(self.compressed_dir / f"{input_file.stem}_{profile_name}.mp4")
        
        job_id = f"sync_job_{int(time.time() * 1000)}"
        job = CompressionJob(
            job_id=job_id,
            input_path=input_path,
            output_path=output_path,
            profile=profile
        )
        
        # Process job immediately
        self._process_compression_job(job)
        
        return {
            "job_id": job.job_id,
            "status": job.status,
            "output_path": job.output_path if job.status == "completed" else None,
            "compression_ratio": job.compression_ratio,
            "original_size_mb": job.original_size_mb,
            "compressed_size_mb": job.compressed_size_mb,
            "error_message": job.error_message
        }
    
    def get_video_info(self, file_path: str) -> VideoInfo:
        """Get detailed video information using FFprobe"""
        try:
            # Use FFprobe to get video information
            cmd = [
                "ffprobe", "-v", "quiet", "-print_format", "json",
                "-show_format", "-show_streams", file_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"FFprobe failed: {result.stderr}")
            
            data = json.loads(result.stdout)
            
            # Extract video stream info
            video_stream = None
            for stream in data["streams"]:
                if stream["codec_type"] == "video":
                    video_stream = stream
                    break
            
            if not video_stream:
                raise RuntimeError("No video stream found")
            
            format_info = data["format"]
            
            # Calculate file checksum
            checksum = self._calculate_file_checksum(file_path)
            
            return VideoInfo(
                file_path=file_path,
                file_size_mb=float(format_info["size"]) / 1024 / 1024,
                duration_seconds=float(format_info["duration"]),
                resolution=f"{video_stream['width']}x{video_stream['height']}",
                frame_rate=eval(video_stream["r_frame_rate"]),  # e.g., "30/1" -> 30.0
                codec=video_stream["codec_name"],
                bitrate=int(format_info.get("bit_rate", 0)),
                creation_time=datetime.fromtimestamp(os.path.getctime(file_path)),
                checksum=checksum
            )
            
        except Exception as e:
            logger.error(f"Error getting video info for {file_path}: {e}")
            raise
    
    def _calculate_file_checksum(self, file_path: str) -> str:
        """Calculate SHA-256 checksum of file"""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def _save_job_to_database(self, job: CompressionJob):
        """Save compression job to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO compression_jobs 
                    (job_id, input_path, output_path, profile_name, status)
                    VALUES (?, ?, ?, ?, ?)
                """, (job.job_id, job.input_path, job.output_path, job.profile.name, job.status))
                conn.commit()
        except Exception as e:
            logger.error(f"Error saving job to database: {e}")
    
    def _update_job_in_database(self, job: CompressionJob):
        """Update compression job in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    UPDATE compression_jobs SET
                    status = ?, progress = ?, start_time = ?, end_time = ?,
                    error_message = ?, original_size_mb = ?, compressed_size_mb = ?,
                    compression_ratio = ?
                    WHERE job_id = ?
                """, (
                    job.status, job.progress,
                    job.start_time.isoformat() if job.start_time else None,
                    job.end_time.isoformat() if job.end_time else None,
                    job.error_message, job.original_size_mb, job.compressed_size_mb,
                    job.compression_ratio, job.job_id
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Error updating job in database: {e}")
    
    async def _get_storage_usage(self) -> Dict[str, float]:
        """Get storage usage statistics"""
        try:
            total, used, free = shutil.disk_usage(self.base_storage_dir)
            
            return {
                "total_gb": total / 1024**3,
                "used_gb": used / 1024**3,
                "free_gb": free / 1024**3,
                "usage_percent": (used / total) * 100
            }
        except Exception as e:
            logger.error(f"Error getting storage usage: {e}")
            return {"total_gb": 0, "used_gb": 0, "free_gb": 0, "usage_percent": 0}
    
    def get_compression_statistics(self) -> Dict[str, Any]:
        """Get comprehensive compression statistics"""
        return {
            "processing_stats": self.stats.copy(),
            "active_jobs": len(self.active_jobs),
            "queued_jobs": len(self.job_queue),
            "profiles": {name: profile.description for name, profile in self.profiles.items()},
            "storage": asyncio.create_task(self._get_storage_usage()) if asyncio.get_event_loop().is_running() else {"error": "No event loop"},
            "configuration": {
                "max_concurrent_jobs": self.max_concurrent_jobs,
                "auto_cleanup_enabled": self.auto_cleanup_enabled,
                "cleanup_older_than_days": self.cleanup_older_than_days,
                "max_storage_gb": self.max_storage_gb
            }
        }
    
    async def shutdown(self):
        """Shutdown the compression service"""
        logger.info("🛑 Shutting down Video Compression Service")
        
        self.processing_active = False
        
        # Cancel background tasks
        if self.cleanup_task:
            self.cleanup_task.cancel()
        if self.monitor_task:
            self.monitor_task.cancel()
        
        # Wait for active jobs to complete (with timeout)
        if self.active_jobs:
            logger.info(f"Waiting for {len(self.active_jobs)} active jobs to complete...")
            self.executor.shutdown(wait=True, timeout=60)
        
        logger.info("✅ Video Compression Service shutdown complete")

# Create singleton instance
video_compression_service = VideoCompressionService() 