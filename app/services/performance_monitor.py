"""
Performance Monitor Service - Real-time system metrics and health monitoring
"""
import asyncio
import psutil
import time
import logging
import json
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque, defaultdict
import threading
from statistics import mean, median
import gc

logger = logging.getLogger(__name__)

@dataclass
class MetricPoint:
    """Single metric data point"""
    timestamp: datetime
    value: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SystemMetrics:
    """System-wide performance metrics"""
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    disk_percent: float
    disk_free_gb: float
    network_bytes_sent: int
    network_bytes_recv: int
    active_processes: int
    load_average: List[float]
    uptime_seconds: float

@dataclass
class SessionMetrics:
    """Session-specific performance metrics"""
    session_id: str
    frame_rate: float
    processing_latency_ms: float
    memory_usage_mb: float
    error_rate: float
    connection_quality: float
    last_activity: datetime
    total_frames: int
    dropped_frames: int

@dataclass
class CameraMetrics:
    """Camera-specific performance metrics"""
    camera_id: str
    camera_type: str
    resolution: str
    fps: float
    frame_drops: int
    connection_stability: float
    processing_time_ms: float
    error_count: int

class PerformanceMonitor:
    """Comprehensive performance monitoring system"""
    
    def __init__(self):
        self.monitoring_active = False
        self.monitoring_interval = 5  # seconds
        self.metrics_retention_hours = 24
        self.max_metrics_per_type = 1000
        
        # Metrics storage
        self.system_metrics_history: deque = deque(maxlen=self.max_metrics_per_type)
        self.session_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.camera_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Real-time metrics
        self.current_system_metrics: Optional[SystemMetrics] = None
        self.current_session_metrics: Dict[str, SessionMetrics] = {}
        self.current_camera_metrics: Dict[str, CameraMetrics] = {}
        
        # Performance thresholds
        self.thresholds = {
            "cpu_critical": 90.0,
            "cpu_warning": 75.0,
            "memory_critical": 90.0,
            "memory_warning": 80.0,
            "disk_critical": 95.0,
            "disk_warning": 85.0,
            "frame_rate_min": 15.0,
            "latency_max_ms": 100.0,
            "error_rate_max": 0.05  # 5%
        }
        
        # Alerting
        self.alert_callbacks: List[Callable] = []
        self.active_alerts: Dict[str, Dict] = {}
        
        # Background tasks
        self.monitor_task: Optional[asyncio.Task] = None
        self.cleanup_task: Optional[asyncio.Task] = None
        
        # Process tracking
        self.process = psutil.Process()
        self.start_time = time.time()
        
        # Statistics
        self.performance_stats = {
            "metrics_collected": 0,
            "alerts_triggered": 0,
            "monitoring_uptime": 0
        }
        
        logger.info("📊 Performance Monitor initialized")
    
    def add_alert_callback(self, callback: Callable):
        """Add callback for performance alerts"""
        self.alert_callbacks.append(callback)
        logger.info("Alert callback registered")
    
    def set_thresholds(self, **kwargs):
        """Update performance thresholds"""
        for key, value in kwargs.items():
            if key in self.thresholds:
                self.thresholds[key] = value
                logger.info(f"Threshold updated: {key} = {value}")
    
    async def start_monitoring(self):
        """Start continuous performance monitoring"""
        if self.monitoring_active:
            logger.warning("Performance monitoring already active")
            return
        
        self.monitoring_active = True
        logger.info("🚀 Starting performance monitoring")
        
        # Start background tasks
        self.monitor_task = asyncio.create_task(self._monitoring_loop())
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        logger.info("✅ Performance monitoring started")
    
    async def stop_monitoring(self):
        """Stop performance monitoring"""
        logger.info("🛑 Stopping performance monitoring")
        
        self.monitoring_active = False
        
        if self.monitor_task:
            self.monitor_task.cancel()
        if self.cleanup_task:
            self.cleanup_task.cancel()
        
        logger.info("✅ Performance monitoring stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                start_time = time.time()
                
                # Collect system metrics
                await self._collect_system_metrics()
                
                # Check thresholds and trigger alerts
                await self._check_thresholds()
                
                # Update statistics
                self.performance_stats["metrics_collected"] += 1
                self.performance_stats["monitoring_uptime"] = time.time() - self.start_time
                
                # Calculate sleep time to maintain interval
                processing_time = time.time() - start_time
                sleep_time = max(0, self.monitoring_interval - processing_time)
                
                await asyncio.sleep(sleep_time)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(self.monitoring_interval)
    
    async def _cleanup_loop(self):
        """Clean up old metrics"""
        while self.monitoring_active:
            try:
                await asyncio.sleep(3600)  # Run every hour
                
                cutoff_time = datetime.now() - timedelta(hours=self.metrics_retention_hours)
                
                # Clean up system metrics
                while (self.system_metrics_history and 
                       self.system_metrics_history[0].timestamp < cutoff_time):
                    self.system_metrics_history.popleft()
                
                # Clean up session metrics
                for session_id in list(self.session_metrics.keys()):
                    metrics = self.session_metrics[session_id]
                    while metrics and metrics[0].timestamp < cutoff_time:
                        metrics.popleft()
                    
                    # Remove empty session metrics
                    if not metrics:
                        del self.session_metrics[session_id]
                
                # Clean up camera metrics
                for camera_id in list(self.camera_metrics.keys()):
                    metrics = self.camera_metrics[camera_id]
                    while metrics and metrics[0].timestamp < cutoff_time:
                        metrics.popleft()
                    
                    # Remove empty camera metrics
                    if not metrics:
                        del self.camera_metrics[camera_id]
                
                # Force garbage collection
                gc.collect()
                
                logger.info("🧹 Performance metrics cleanup completed")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
    
    async def _collect_system_metrics(self):
        """Collect system-wide performance metrics"""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            
            # Network metrics
            net_io = psutil.net_io_counters()
            
            # Process metrics
            process_count = len(psutil.pids())
            
            # Load average (Unix-like systems)
            try:
                load_avg = list(psutil.getloadavg())
            except (AttributeError, OSError):
                load_avg = [0.0, 0.0, 0.0]
            
            # Create metrics object
            metrics = SystemMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_used_mb=memory.used / 1024 / 1024,
                memory_available_mb=memory.available / 1024 / 1024,
                disk_percent=disk.percent,
                disk_free_gb=disk.free / 1024 / 1024 / 1024,
                network_bytes_sent=net_io.bytes_sent,
                network_bytes_recv=net_io.bytes_recv,
                active_processes=process_count,
                load_average=load_avg,
                uptime_seconds=time.time() - self.start_time
            )
            
            # Store current metrics
            self.current_system_metrics = metrics
            
            # Add to history
            metric_point = MetricPoint(
                timestamp=datetime.now(),
                value=0,  # Not used for system metrics
                metadata=metrics.__dict__
            )
            self.system_metrics_history.append(metric_point)
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
    
    async def _check_thresholds(self):
        """Check performance thresholds and trigger alerts"""
        if not self.current_system_metrics:
            return
        
        metrics = self.current_system_metrics
        alerts_to_trigger = []
        
        # CPU threshold check
        if metrics.cpu_percent >= self.thresholds["cpu_critical"]:
            alerts_to_trigger.append({
                "type": "cpu_critical",
                "message": f"Critical CPU usage: {metrics.cpu_percent:.1f}%",
                "value": metrics.cpu_percent,
                "threshold": self.thresholds["cpu_critical"]
            })
        elif metrics.cpu_percent >= self.thresholds["cpu_warning"]:
            alerts_to_trigger.append({
                "type": "cpu_warning",
                "message": f"High CPU usage: {metrics.cpu_percent:.1f}%",
                "value": metrics.cpu_percent,
                "threshold": self.thresholds["cpu_warning"]
            })
        
        # Memory threshold check
        if metrics.memory_percent >= self.thresholds["memory_critical"]:
            alerts_to_trigger.append({
                "type": "memory_critical",
                "message": f"Critical memory usage: {metrics.memory_percent:.1f}%",
                "value": metrics.memory_percent,
                "threshold": self.thresholds["memory_critical"]
            })
        elif metrics.memory_percent >= self.thresholds["memory_warning"]:
            alerts_to_trigger.append({
                "type": "memory_warning",
                "message": f"High memory usage: {metrics.memory_percent:.1f}%",
                "value": metrics.memory_percent,
                "threshold": self.thresholds["memory_warning"]
            })
        
        # Disk threshold check
        if metrics.disk_percent >= self.thresholds["disk_critical"]:
            alerts_to_trigger.append({
                "type": "disk_critical",
                "message": f"Critical disk usage: {metrics.disk_percent:.1f}%",
                "value": metrics.disk_percent,
                "threshold": self.thresholds["disk_critical"]
            })
        elif metrics.disk_percent >= self.thresholds["disk_warning"]:
            alerts_to_trigger.append({
                "type": "disk_warning",
                "message": f"High disk usage: {metrics.disk_percent:.1f}%",
                "value": metrics.disk_percent,
                "threshold": self.thresholds["disk_warning"]
            })
        
        # Trigger alerts
        for alert in alerts_to_trigger:
            await self._trigger_alert(alert)
    
    async def _trigger_alert(self, alert_data: Dict):
        """Trigger a performance alert"""
        alert_id = f"{alert_data['type']}_{int(time.time())}"
        
        # Check if similar alert is already active (debouncing)
        for active_id, active_alert in self.active_alerts.items():
            if (active_alert["type"] == alert_data["type"] and 
                time.time() - active_alert["triggered_at"] < 300):  # 5 minutes
                return  # Don't trigger duplicate alert
        
        # Add timestamp and ID
        alert_data["alert_id"] = alert_id
        alert_data["triggered_at"] = time.time()
        
        # Store active alert
        self.active_alerts[alert_id] = alert_data
        
        # Update statistics
        self.performance_stats["alerts_triggered"] += 1
        
        # Call alert callbacks
        for callback in self.alert_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(alert_data)
                else:
                    callback(alert_data)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")
        
        logger.warning(f"🚨 Performance Alert: {alert_data['message']}")
    
    def record_session_metric(self, session_id: str, **kwargs):
        """Record session-specific metrics"""
        try:
            current_time = datetime.now()
            
            # Get or create session metrics
            if session_id not in self.current_session_metrics:
                self.current_session_metrics[session_id] = SessionMetrics(
                    session_id=session_id,
                    frame_rate=0.0,
                    processing_latency_ms=0.0,
                    memory_usage_mb=0.0,
                    error_rate=0.0,
                    connection_quality=1.0,
                    last_activity=current_time,
                    total_frames=0,
                    dropped_frames=0
                )
            
            metrics = self.current_session_metrics[session_id]
            
            # Update metrics with provided values
            for key, value in kwargs.items():
                if hasattr(metrics, key):
                    setattr(metrics, key, value)
            
            metrics.last_activity = current_time
            
            # Store in history
            metric_point = MetricPoint(
                timestamp=current_time,
                value=0,
                metadata=metrics.__dict__.copy()
            )
            self.session_metrics[session_id].append(metric_point)
            
        except Exception as e:
            logger.error(f"Error recording session metric: {e}")
    
    def record_camera_metric(self, camera_id: str, camera_type: str, **kwargs):
        """Record camera-specific metrics"""
        try:
            current_time = datetime.now()
            
            # Get or create camera metrics
            if camera_id not in self.current_camera_metrics:
                self.current_camera_metrics[camera_id] = CameraMetrics(
                    camera_id=camera_id,
                    camera_type=camera_type,
                    resolution="unknown",
                    fps=0.0,
                    frame_drops=0,
                    connection_stability=1.0,
                    processing_time_ms=0.0,
                    error_count=0
                )
            
            metrics = self.current_camera_metrics[camera_id]
            
            # Update metrics with provided values
            for key, value in kwargs.items():
                if hasattr(metrics, key):
                    setattr(metrics, key, value)
            
            # Store in history
            metric_point = MetricPoint(
                timestamp=current_time,
                value=0,
                metadata=metrics.__dict__.copy()
            )
            self.camera_metrics[camera_id].append(metric_point)
            
        except Exception as e:
            logger.error(f"Error recording camera metric: {e}")
    
    def get_system_metrics(self) -> Optional[SystemMetrics]:
        """Get current system metrics"""
        return self.current_system_metrics
    
    def get_session_metrics(self, session_id: Optional[str] = None) -> Dict:
        """Get session metrics"""
        if session_id:
            return self.current_session_metrics.get(session_id)
        return self.current_session_metrics.copy()
    
    def get_camera_metrics(self, camera_id: Optional[str] = None) -> Dict:
        """Get camera metrics"""
        if camera_id:
            return self.current_camera_metrics.get(camera_id)
        return self.current_camera_metrics.copy()
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        try:
            system_health = "healthy"
            if self.current_system_metrics:
                if (self.current_system_metrics.cpu_percent > self.thresholds["cpu_warning"] or
                    self.current_system_metrics.memory_percent > self.thresholds["memory_warning"] or
                    self.current_system_metrics.disk_percent > self.thresholds["disk_warning"]):
                    system_health = "warning"
                
                if (self.current_system_metrics.cpu_percent > self.thresholds["cpu_critical"] or
                    self.current_system_metrics.memory_percent > self.thresholds["memory_critical"] or
                    self.current_system_metrics.disk_percent > self.thresholds["disk_critical"]):
                    system_health = "critical"
            
            return {
                "system": {
                    "health": system_health,
                    "metrics": self.current_system_metrics.__dict__ if self.current_system_metrics else None,
                    "thresholds": self.thresholds
                },
                "sessions": {
                    "active_count": len(self.current_session_metrics),
                    "metrics": {sid: metrics.__dict__ for sid, metrics in self.current_session_metrics.items()}
                },
                "cameras": {
                    "active_count": len(self.current_camera_metrics),
                    "metrics": {cid: metrics.__dict__ for cid, metrics in self.current_camera_metrics.items()}
                },
                "alerts": {
                    "active_count": len(self.active_alerts),
                    "active_alerts": list(self.active_alerts.values())
                },
                "statistics": self.performance_stats,
                "monitoring": {
                    "active": self.monitoring_active,
                    "interval": self.monitoring_interval,
                    "retention_hours": self.metrics_retention_hours
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating performance summary: {e}")
            return {"error": str(e)}
    
    def get_historical_data(self, 
                          metric_type: str = "system", 
                          identifier: Optional[str] = None, 
                          hours: int = 1) -> List[Dict]:
        """Get historical performance data"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            if metric_type == "system":
                return [
                    {
                        "timestamp": point.timestamp.isoformat(),
                        "data": point.metadata
                    }
                    for point in self.system_metrics_history
                    if point.timestamp >= cutoff_time
                ]
            elif metric_type == "session" and identifier:
                if identifier in self.session_metrics:
                    return [
                        {
                            "timestamp": point.timestamp.isoformat(),
                            "data": point.metadata
                        }
                        for point in self.session_metrics[identifier]
                        if point.timestamp >= cutoff_time
                    ]
            elif metric_type == "camera" and identifier:
                if identifier in self.camera_metrics:
                    return [
                        {
                            "timestamp": point.timestamp.isoformat(),
                            "data": point.metadata
                        }
                        for point in self.camera_metrics[identifier]
                        if point.timestamp >= cutoff_time
                    ]
            
            return []
            
        except Exception as e:
            logger.error(f"Error getting historical data: {e}")
            return []
    
    def clear_metrics(self, metric_type: Optional[str] = None):
        """Clear stored metrics"""
        if metric_type == "system" or metric_type is None:
            self.system_metrics_history.clear()
            self.current_system_metrics = None
        
        if metric_type == "session" or metric_type is None:
            self.session_metrics.clear()
            self.current_session_metrics.clear()
        
        if metric_type == "camera" or metric_type is None:
            self.camera_metrics.clear()
            self.current_camera_metrics.clear()
        
        if metric_type == "alerts" or metric_type is None:
            self.active_alerts.clear()
        
        logger.info(f"Cleared {metric_type or 'all'} metrics")

# Create singleton instance
performance_monitor = PerformanceMonitor() 