"""
Automated Backup Service - Comprehensive backup and recovery system
"""
import os
import asyncio
import logging
import shutil
import zipfile
import json
import hashlib
import time
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from pathlib import Path
import sqlite3
import boto3
from botocore.exceptions import NoCredentialsError, ClientError
import threading
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

@dataclass
class BackupItem:
    """Individual backup item"""
    item_id: str
    source_path: str
    backup_path: str
    item_type: str  # video, session_data, config, database
    size_bytes: int
    checksum: str
    created_at: datetime
    last_verified: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class BackupJob:
    """Backup job configuration"""
    job_id: str
    name: str
    source_paths: List[str]
    destination: str
    backup_type: str  # incremental, full, differential
    compression: bool = True
    encryption: bool = False
    retention_days: int = 30
    schedule: Optional[str] = None  # cron-like schedule
    enabled: bool = True
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class BackupDestination:
    """Backup destination configuration"""
    dest_id: str
    name: str
    dest_type: str  # local, s3, ftp, network
    path: str
    credentials: Dict[str, str] = field(default_factory=dict)
    available: bool = True
    last_check: Optional[datetime] = None
    capacity_gb: Optional[float] = None
    used_gb: Optional[float] = None

class BackupService:
    """Comprehensive automated backup service"""
    
    def __init__(self, base_data_dir: str = "/data"):
        self.base_data_dir = Path(base_data_dir)
        self.backup_dir = self.base_data_dir / "backups"
        self.temp_dir = self.base_data_dir / "temp_backup"
        
        # Create directories
        for directory in [self.backup_dir, self.temp_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Database for backup tracking
        self.db_path = self.backup_dir / "backup_db.sqlite"
        self._init_database()
        
        # Configuration
        self.max_concurrent_backups = 2
        self.compression_level = 6  # 0-9, higher = better compression
        self.verification_enabled = True
        self.auto_cleanup_enabled = True
        self.max_backup_size_gb = 100
        
        # Runtime state
        self.backup_jobs: Dict[str, BackupJob] = {}
        self.destinations: Dict[str, BackupDestination] = {}
        self.active_backups: Dict[str, Any] = {}
        self.backup_queue: List[str] = []
        
        # Threading
        self.executor = ThreadPoolExecutor(max_workers=self.max_concurrent_backups)
        self.scheduler_active = False
        
        # Background tasks
        self.scheduler_task = None
        self.cleanup_task = None
        self.verification_task = None
        
        # Statistics
        self.stats = {
            "total_backups_created": 0,
            "total_data_backed_up_gb": 0.0,
            "successful_backups": 0,
            "failed_backups": 0,
            "total_space_used_gb": 0.0,
            "last_backup_time": None,
            "average_backup_duration_minutes": 0.0
        }
        
        # Event callbacks
        self.backup_callbacks: List[Callable] = []
        
        logger.info("💾 Backup Service initialized")
        
        # Initialize default configurations
        self._create_default_jobs()
        self._create_default_destinations()
        
        # Start background services
        self._start_background_tasks()
    
    def _init_database(self):
        """Initialize SQLite database for backup tracking"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS backup_items (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        item_id TEXT UNIQUE NOT NULL,
                        source_path TEXT NOT NULL,
                        backup_path TEXT NOT NULL,
                        item_type TEXT NOT NULL,
                        size_bytes INTEGER NOT NULL,
                        checksum TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        last_verified TEXT,
                        metadata TEXT
                    )
                """)
                
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS backup_jobs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        job_id TEXT UNIQUE NOT NULL,
                        name TEXT NOT NULL,
                        source_paths TEXT NOT NULL,
                        destination TEXT NOT NULL,
                        backup_type TEXT NOT NULL,
                        compression BOOLEAN NOT NULL,
                        encryption BOOLEAN NOT NULL,
                        retention_days INTEGER NOT NULL,
                        schedule TEXT,
                        enabled BOOLEAN NOT NULL,
                        last_run TEXT,
                        next_run TEXT,
                        metadata TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS backup_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        job_id TEXT NOT NULL,
                        start_time TEXT NOT NULL,
                        end_time TEXT,
                        status TEXT NOT NULL,
                        items_backed_up INTEGER DEFAULT 0,
                        total_size_mb REAL DEFAULT 0.0,
                        error_message TEXT,
                        backup_path TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                conn.commit()
                logger.info("📚 Backup database initialized")
        except Exception as e:
            logger.error(f"Failed to initialize backup database: {e}")
    
    def _create_default_jobs(self):
        """Create default backup jobs"""
        default_jobs = [
            BackupJob(
                job_id="daily_videos",
                name="Daily Video Backup",
                source_paths=[str(self.base_data_dir / "videos")],
                destination="local_primary",
                backup_type="incremental",
                compression=True,
                retention_days=30,
                schedule="0 2 * * *"  # Daily at 2 AM
            ),
            BackupJob(
                job_id="session_data",
                name="Session Data Backup",
                source_paths=[str(self.base_data_dir / "sessions")],
                destination="local_primary",
                backup_type="incremental",
                compression=True,
                retention_days=14,
                schedule="0 */6 * * *"  # Every 6 hours
            ),
            BackupJob(
                job_id="configuration",
                name="Configuration Backup",
                source_paths=["/app/config", str(self.base_data_dir / "logs")],
                destination="local_primary",
                backup_type="full",
                compression=True,
                retention_days=60,
                schedule="0 3 * * 0"  # Weekly on Sunday at 3 AM
            )
        ]
        
        for job in default_jobs:
            self.backup_jobs[job.job_id] = job
            self._save_job_to_database(job)
    
    def _create_default_destinations(self):
        """Create default backup destinations"""
        default_destinations = [
            BackupDestination(
                dest_id="local_primary",
                name="Local Primary Backup",
                dest_type="local",
                path=str(self.backup_dir / "primary"),
                capacity_gb=50.0
            ),
            BackupDestination(
                dest_id="local_archive",
                name="Local Archive Backup",
                dest_type="local",
                path=str(self.backup_dir / "archive"),
                capacity_gb=100.0
            )
        ]
        
        for dest in default_destinations:
            self.destinations[dest.dest_id] = dest
            # Create destination directory
            Path(dest.path).mkdir(parents=True, exist_ok=True)
    
    def _start_background_tasks(self):
        """Start background backup tasks"""
        self.scheduler_active = True
        self.scheduler_task = asyncio.create_task(self._scheduler_loop())
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
        self.verification_task = asyncio.create_task(self._verification_loop())
        logger.info("🚀 Backup background tasks started")
    
    async def _scheduler_loop(self):
        """Background scheduler for automated backups"""
        while self.scheduler_active:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                current_time = datetime.now()
                
                for job_id, job in self.backup_jobs.items():
                    if not job.enabled:
                        continue
                    
                    # Check if job should run
                    should_run = False
                    
                    if job.schedule and self._should_run_scheduled_job(job, current_time):
                        should_run = True
                    elif job.last_run is None:  # Never run before
                        should_run = True
                    
                    if should_run and job_id not in self.active_backups:
                        logger.info(f"📅 Scheduling backup job: {job.name}")
                        await self.run_backup_job_async(job_id)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in backup scheduler: {e}")
    
    async def _cleanup_loop(self):
        """Background cleanup of old backups"""
        while self.scheduler_active:
            try:
                await asyncio.sleep(3600)  # Run every hour
                
                if self.auto_cleanup_enabled:
                    await self._perform_cleanup()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in backup cleanup: {e}")
    
    async def _verification_loop(self):
        """Background verification of backup integrity"""
        while self.scheduler_active:
            try:
                await asyncio.sleep(86400)  # Run daily
                
                if self.verification_enabled:
                    await self._verify_backup_integrity()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in backup verification: {e}")
    
    def _should_run_scheduled_job(self, job: BackupJob, current_time: datetime) -> bool:
        """Check if scheduled job should run based on cron-like schedule"""
        if not job.schedule:
            return False
        
        # Simple cron parser for basic schedules
        # Format: "minute hour day_of_month month day_of_week"
        try:
            parts = job.schedule.split()
            if len(parts) != 5:
                return False
            
            minute, hour, day, month, dow = parts
            
            # Check if current time matches schedule
            if minute != "*" and int(minute) != current_time.minute:
                return False
            if hour != "*" and int(hour) != current_time.hour:
                return False
            if day != "*" and int(day) != current_time.day:
                return False
            if month != "*" and int(month) != current_time.month:
                return False
            if dow != "*" and int(dow) != current_time.weekday():
                return False
            
            # Check if job ran recently (avoid duplicate runs)
            if job.last_run:
                time_since_last = current_time - job.last_run
                if time_since_last.total_seconds() < 3600:  # 1 hour minimum between runs
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error parsing schedule for job {job.job_id}: {e}")
            return False
    
    async def run_backup_job_async(self, job_id: str) -> str:
        """Run backup job asynchronously"""
        if job_id not in self.backup_jobs:
            raise ValueError(f"Backup job not found: {job_id}")
        
        job = self.backup_jobs[job_id]
        
        if job_id in self.active_backups:
            raise RuntimeError(f"Backup job already running: {job_id}")
        
        # Create backup run ID
        run_id = f"{job_id}_{int(time.time())}"
        
        # Mark as active
        self.active_backups[job_id] = {
            "run_id": run_id,
            "start_time": datetime.now(),
            "status": "running"
        }
        
        # Start backup in thread pool
        loop = asyncio.get_event_loop()
        future = loop.run_in_executor(self.executor, self._execute_backup_job, job, run_id)
        
        logger.info(f"🚀 Started backup job: {job.name} (ID: {run_id})")
        return run_id
    
    def _execute_backup_job(self, job: BackupJob, run_id: str):
        """Execute backup job in thread"""
        start_time = datetime.now()
        backup_path = None
        items_backed_up = 0
        total_size_mb = 0.0
        error_message = None
        
        try:
            logger.info(f"📦 Executing backup job: {job.name}")
            
            # Get destination
            if job.destination not in self.destinations:
                raise ValueError(f"Destination not found: {job.destination}")
            
            destination = self.destinations[job.destination]
            
            # Create backup directory
            backup_timestamp = start_time.strftime("%Y%m%d_%H%M%S")
            backup_path = Path(destination.path) / f"{job.job_id}_{backup_timestamp}"
            backup_path.mkdir(parents=True, exist_ok=True)
            
            # Process each source path
            for source_path in job.source_paths:
                source = Path(source_path)
                if not source.exists():
                    logger.warning(f"Source path does not exist: {source_path}")
                    continue
                
                if source.is_file():
                    # Backup single file
                    success, size = self._backup_file(source, backup_path, job)
                    if success:
                        items_backed_up += 1
                        total_size_mb += size
                elif source.is_dir():
                    # Backup directory
                    backed_up, size = self._backup_directory(source, backup_path, job)
                    items_backed_up += backed_up
                    total_size_mb += size
            
            # Create backup manifest
            self._create_backup_manifest(backup_path, job, items_backed_up, total_size_mb)
            
            # Update job statistics
            job.last_run = datetime.now()
            self.stats["successful_backups"] += 1
            self.stats["total_backups_created"] += 1
            self.stats["total_data_backed_up_gb"] += total_size_mb / 1024
            
            logger.info(f"✅ Backup completed: {job.name} ({items_backed_up} items, {total_size_mb:.1f}MB)")
            
        except Exception as e:
            error_message = str(e)
            self.stats["failed_backups"] += 1
            logger.error(f"❌ Backup failed: {job.name} - {error_message}")
            
        finally:
            # Record backup history
            end_time = datetime.now()
            duration_minutes = (end_time - start_time).total_seconds() / 60
            
            # Update average duration
            if self.stats["total_backups_created"] > 0:
                current_avg = self.stats["average_backup_duration_minutes"]
                total_backups = self.stats["total_backups_created"]
                self.stats["average_backup_duration_minutes"] = (
                    (current_avg * (total_backups - 1) + duration_minutes) / total_backups
                )
            
            self._save_backup_history(job.job_id, start_time, end_time, 
                                    "success" if error_message is None else "failed",
                                    items_backed_up, total_size_mb, error_message, str(backup_path))
            
            # Remove from active backups
            if job.job_id in self.active_backups:
                del self.active_backups[job.job_id]
            
            # Trigger callbacks
            for callback in self.backup_callbacks:
                try:
                    callback({
                        "job_id": job.job_id,
                        "job_name": job.name,
                        "status": "success" if error_message is None else "failed",
                        "items_backed_up": items_backed_up,
                        "total_size_mb": total_size_mb,
                        "duration_minutes": duration_minutes,
                        "error_message": error_message
                    })
                except Exception as callback_error:
                    logger.error(f"Backup callback error: {callback_error}")
    
    def _backup_file(self, source_file: Path, backup_dir: Path, job: BackupJob) -> Tuple[bool, float]:
        """Backup a single file"""
        try:
            # Calculate relative path to preserve directory structure
            relative_path = source_file.name
            target_path = backup_dir / relative_path
            
            # Create target directory if needed
            target_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy file
            if job.compression and source_file.suffix.lower() not in ['.zip', '.gz', '.bz2', '.7z']:
                # Compress file
                compressed_path = target_path.with_suffix(target_path.suffix + '.gz')
                with open(source_file, 'rb') as f_in:
                    import gzip
                    with gzip.open(compressed_path, 'wb', compresslevel=self.compression_level) as f_out:
                        shutil.copyfileobj(f_in, f_out)
                target_path = compressed_path
            else:
                # Copy without compression
                shutil.copy2(source_file, target_path)
            
            # Calculate size and checksum
            size_mb = target_path.stat().st_size / 1024 / 1024
            checksum = self._calculate_checksum(target_path)
            
            # Create backup item record
            item = BackupItem(
                item_id=f"{job.job_id}_{source_file.name}_{int(time.time())}",
                source_path=str(source_file),
                backup_path=str(target_path),
                item_type="file",
                size_bytes=target_path.stat().st_size,
                checksum=checksum,
                created_at=datetime.now()
            )
            
            self._save_backup_item(item)
            return True, size_mb
            
        except Exception as e:
            logger.error(f"Failed to backup file {source_file}: {e}")
            return False, 0.0
    
    def _backup_directory(self, source_dir: Path, backup_dir: Path, job: BackupJob) -> Tuple[int, float]:
        """Backup a directory recursively"""
        items_backed_up = 0
        total_size_mb = 0.0
        
        try:
            # Create archive of directory
            archive_name = f"{source_dir.name}.zip"
            archive_path = backup_dir / archive_name
            
            with zipfile.ZipFile(archive_path, 'w', 
                               zipfile.ZIP_DEFLATED if job.compression else zipfile.ZIP_STORED,
                               compresslevel=self.compression_level if job.compression else None) as zipf:
                
                for file_path in source_dir.rglob('*'):
                    if file_path.is_file():
                        # Add file to archive with relative path
                        arc_name = file_path.relative_to(source_dir)
                        zipf.write(file_path, arc_name)
                        items_backed_up += 1
            
            # Calculate archive size and checksum
            total_size_mb = archive_path.stat().st_size / 1024 / 1024
            checksum = self._calculate_checksum(archive_path)
            
            # Create backup item record
            item = BackupItem(
                item_id=f"{job.job_id}_{source_dir.name}_{int(time.time())}",
                source_path=str(source_dir),
                backup_path=str(archive_path),
                item_type="directory",
                size_bytes=archive_path.stat().st_size,
                checksum=checksum,
                created_at=datetime.now(),
                metadata={"files_count": items_backed_up}
            )
            
            self._save_backup_item(item)
            
        except Exception as e:
            logger.error(f"Failed to backup directory {source_dir}: {e}")
            
        return items_backed_up, total_size_mb
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA-256 checksum of file"""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def _create_backup_manifest(self, backup_path: Path, job: BackupJob, 
                              items_count: int, total_size_mb: float):
        """Create backup manifest file"""
        manifest = {
            "job_id": job.job_id,
            "job_name": job.name,
            "backup_type": job.backup_type,
            "created_at": datetime.now().isoformat(),
            "items_count": items_count,
            "total_size_mb": total_size_mb,
            "compression_enabled": job.compression,
            "encryption_enabled": job.encryption,
            "source_paths": job.source_paths,
            "backup_version": "1.0"
        }
        
        manifest_path = backup_path / "backup_manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
    
    def _save_job_to_database(self, job: BackupJob):
        """Save backup job to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO backup_jobs 
                    (job_id, name, source_paths, destination, backup_type, compression, 
                     encryption, retention_days, schedule, enabled, last_run, next_run, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    job.job_id, job.name, json.dumps(job.source_paths), job.destination,
                    job.backup_type, job.compression, job.encryption, job.retention_days,
                    job.schedule, job.enabled,
                    job.last_run.isoformat() if job.last_run else None,
                    job.next_run.isoformat() if job.next_run else None,
                    json.dumps(job.metadata)
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Error saving backup job to database: {e}")
    
    def _save_backup_item(self, item: BackupItem):
        """Save backup item to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO backup_items 
                    (item_id, source_path, backup_path, item_type, size_bytes, 
                     checksum, created_at, last_verified, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    item.item_id, item.source_path, item.backup_path, item.item_type,
                    item.size_bytes, item.checksum, item.created_at.isoformat(),
                    item.last_verified.isoformat() if item.last_verified else None,
                    json.dumps(item.metadata)
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Error saving backup item to database: {e}")
    
    def _save_backup_history(self, job_id: str, start_time: datetime, end_time: datetime,
                           status: str, items_count: int, total_size_mb: float, 
                           error_message: Optional[str], backup_path: str):
        """Save backup history record"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO backup_history 
                    (job_id, start_time, end_time, status, items_backed_up, 
                     total_size_mb, error_message, backup_path)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    job_id, start_time.isoformat(), end_time.isoformat(), status,
                    items_count, total_size_mb, error_message, backup_path
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Error saving backup history: {e}")
    
    async def _perform_cleanup(self):
        """Perform cleanup of old backups"""
        try:
            cleanup_count = 0
            space_freed_gb = 0.0
            
            for job_id, job in self.backup_jobs.items():
                cutoff_date = datetime.now() - timedelta(days=job.retention_days)
                
                # Get old backup items
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute("""
                        SELECT item_id, backup_path, size_bytes FROM backup_items 
                        WHERE created_at < ? AND item_id LIKE ?
                    """, (cutoff_date.isoformat(), f"{job_id}_%"))
                    
                    old_items = cursor.fetchall()
                
                # Remove old backup files
                for item_id, backup_path, size_bytes in old_items:
                    try:
                        if os.path.exists(backup_path):
                            os.remove(backup_path)
                            space_freed_gb += size_bytes / 1024**3
                            cleanup_count += 1
                        
                        # Remove from database
                        conn.execute("DELETE FROM backup_items WHERE item_id = ?", (item_id,))
                        
                    except Exception as e:
                        logger.error(f"Error removing backup item {item_id}: {e}")
                
                conn.commit()
            
            if cleanup_count > 0:
                logger.info(f"🧹 Backup cleanup: {cleanup_count} items removed, {space_freed_gb:.2f}GB freed")
            
        except Exception as e:
            logger.error(f"Error during backup cleanup: {e}")
    
    async def _verify_backup_integrity(self):
        """Verify integrity of backup files"""
        try:
            verification_count = 0
            errors_found = 0
            
            # Get backup items that haven't been verified recently
            cutoff_date = datetime.now() - timedelta(days=7)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT item_id, backup_path, checksum FROM backup_items 
                    WHERE last_verified IS NULL OR last_verified < ?
                    LIMIT 100
                """, (cutoff_date.isoformat(),))
                
                items_to_verify = cursor.fetchall()
            
            for item_id, backup_path, expected_checksum in items_to_verify:
                try:
                    if not os.path.exists(backup_path):
                        logger.error(f"Backup file missing: {backup_path}")
                        errors_found += 1
                        continue
                    
                    # Calculate current checksum
                    current_checksum = self._calculate_checksum(Path(backup_path))
                    
                    if current_checksum == expected_checksum:
                        # Update last verified time
                        with sqlite3.connect(self.db_path) as conn:
                            conn.execute("""
                                UPDATE backup_items SET last_verified = ? WHERE item_id = ?
                            """, (datetime.now().isoformat(), item_id))
                            conn.commit()
                        verification_count += 1
                    else:
                        logger.error(f"Backup integrity check failed: {backup_path}")
                        errors_found += 1
                        
                except Exception as e:
                    logger.error(f"Error verifying backup {item_id}: {e}")
                    errors_found += 1
            
            if verification_count > 0:
                logger.info(f"🔍 Backup verification: {verification_count} items verified, {errors_found} errors found")
            
        except Exception as e:
            logger.error(f"Error during backup verification: {e}")
    
    def add_backup_callback(self, callback: Callable):
        """Add callback for backup events"""
        self.backup_callbacks.append(callback)
        logger.info("Backup callback registered")
    
    def get_backup_status(self) -> Dict[str, Any]:
        """Get comprehensive backup status"""
        return {
            "statistics": self.stats.copy(),
            "active_backups": len(self.active_backups),
            "configured_jobs": len(self.backup_jobs),
            "destinations": len(self.destinations),
            "scheduler_active": self.scheduler_active,
            "configuration": {
                "max_concurrent_backups": self.max_concurrent_backups,
                "verification_enabled": self.verification_enabled,
                "auto_cleanup_enabled": self.auto_cleanup_enabled,
                "max_backup_size_gb": self.max_backup_size_gb
            }
        }
    
    async def shutdown(self):
        """Shutdown backup service"""
        logger.info("🛑 Shutting down Backup Service")
        
        self.scheduler_active = False
        
        # Cancel background tasks
        if self.scheduler_task:
            self.scheduler_task.cancel()
        if self.cleanup_task:
            self.cleanup_task.cancel()
        if self.verification_task:
            self.verification_task.cancel()
        
        # Wait for active backups to complete
        if self.active_backups:
            logger.info(f"Waiting for {len(self.active_backups)} active backups to complete...")
            self.executor.shutdown(wait=True, timeout=300)  # 5 minutes timeout
        
        logger.info("✅ Backup Service shutdown complete")

# Create singleton instance
backup_service = BackupService() 