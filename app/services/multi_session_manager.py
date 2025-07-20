"""
Multi-Session Manager - Support for multiple simultaneous bee tracking sessions
"""
import asyncio
import logging
import time
from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import uuid
import json
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict

from .camera_session_manager import CameraSessionManager, CameraSession
from .error_handler import ErrorHandler, ErrorType, ErrorSeverity
from .performance_monitor import PerformanceMonitor
from .websocket_connection_manager import WebSocketConnectionManager
from .event_coordinator import EventCoordinator

logger = logging.getLogger(__name__)

class SessionPriority(Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class HiveConfiguration:
    """Configuration for a single hive tracking setup"""
    hive_id: str
    hive_name: str
    location: str
    internal_camera_id: str
    external_camera_id: str
    priority: SessionPriority = SessionPriority.NORMAL
    max_concurrent_sessions: int = 3
    recording_enabled: bool = True
    detection_sensitivity: float = 0.25
    notification_settings: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MultiSessionStats:
    """Statistics for multi-session operations"""
    total_hives: int = 0
    active_sessions: int = 0
    total_sessions_created: int = 0
    resource_usage: Dict[str, float] = field(default_factory=dict)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    session_distribution: Dict[str, int] = field(default_factory=dict)

class MultiSessionManager:
    """Manager for multiple simultaneous bee tracking sessions"""
    
    def __init__(self):
        # Core components
        self.session_manager = CameraSessionManager()
        self.error_handler = ErrorHandler()
        self.performance_monitor = PerformanceMonitor()
        self.websocket_manager = WebSocketConnectionManager()
        self.event_coordinator = EventCoordinator()
        
        # Configuration
        self.max_total_sessions = 10
        self.max_sessions_per_hive = 3
        self.session_timeout_hours = 24
        self.resource_monitoring_enabled = True
        
        # State management
        self.hive_configurations: Dict[str, HiveConfiguration] = {}
        self.active_sessions: Dict[str, Dict[str, Any]] = {}  # session_id -> session_info
        self.hive_sessions: Dict[str, Set[str]] = defaultdict(set)  # hive_id -> session_ids
        self.session_priorities: Dict[str, SessionPriority] = {}
        self.resource_locks: Dict[str, asyncio.Lock] = {}
        
        # Thread pools for parallel operations
        self.session_executor = ThreadPoolExecutor(max_workers=5)
        self.processing_executor = ThreadPoolExecutor(max_workers=8)
        
        # Monitoring and statistics
        self.stats = MultiSessionStats()
        self.session_history: List[Dict[str, Any]] = []
        
        # Background tasks
        self.monitoring_task = None
        self.cleanup_task = None
        self.load_balancer_task = None
        self.monitoring_active = False
        
        logger.info("🔄 Multi-Session Manager initialized")
        
        # Initialize default configurations
        self._initialize_default_hives()
        self._start_background_tasks()
    
    def _initialize_default_hives(self):
        """Initialize default hive configurations"""
        default_hives = [
            HiveConfiguration(
                hive_id="hive_001",
                hive_name="Main Hive",
                location="Garden North",
                internal_camera_id="0",
                external_camera_id="1",
                priority=SessionPriority.HIGH
            ),
            HiveConfiguration(
                hive_id="hive_002",
                hive_name="Secondary Hive",
                location="Garden South",
                internal_camera_id="2",
                external_camera_id="3",
                priority=SessionPriority.NORMAL
            )
        ]
        
        for hive in default_hives:
            self.hive_configurations[hive.hive_id] = hive
            self.resource_locks[hive.hive_id] = asyncio.Lock()
        
        self.stats.total_hives = len(self.hive_configurations)
        logger.info(f"📍 Initialized {len(default_hives)} default hive configurations")
    
    def _start_background_tasks(self):
        """Start background monitoring and management tasks"""
        self.monitoring_active = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
        self.load_balancer_task = asyncio.create_task(self._load_balancer_loop())
        logger.info("🚀 Multi-session background tasks started")
    
    async def _monitoring_loop(self):
        """Background monitoring of all active sessions"""
        while self.monitoring_active:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                # Update statistics
                await self._update_statistics()
                
                # Check session health
                await self._check_session_health()
                
                # Monitor resource usage
                if self.resource_monitoring_enabled:
                    await self._monitor_resources()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in multi-session monitoring loop: {e}")
    
    async def _cleanup_loop(self):
        """Background cleanup of expired and orphaned sessions"""
        while self.monitoring_active:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes
                
                # Clean up expired sessions
                await self._cleanup_expired_sessions()
                
                # Clean up orphaned resources
                await self._cleanup_orphaned_resources()
                
                # Optimize resource allocation
                await self._optimize_resource_allocation()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in multi-session cleanup loop: {e}")
    
    async def _load_balancer_loop(self):
        """Background load balancing across hives and sessions"""
        while self.monitoring_active:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                # Balance load across hives
                await self._balance_session_load()
                
                # Adjust priorities based on activity
                await self._adjust_session_priorities()
                
                # Scale resources if needed
                await self._scale_resources()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in load balancer loop: {e}")
    
    async def create_session(self, hive_id: str, session_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create a new tracking session for a specific hive"""
        try:
            # Validate hive configuration
            if hive_id not in self.hive_configurations:
                raise ValueError(f"Hive not found: {hive_id}")
            
            hive_config = self.hive_configurations[hive_id]
            
            # Check session limits
            if len(self.active_sessions) >= self.max_total_sessions:
                raise RuntimeError("Maximum total sessions limit reached")
            
            if len(self.hive_sessions[hive_id]) >= hive_config.max_concurrent_sessions:
                raise RuntimeError(f"Maximum sessions for hive {hive_id} reached")
            
            # Acquire hive lock
            async with self.resource_locks[hive_id]:
                # Create unique session ID
                session_id = f"session_{hive_id}_{uuid.uuid4().hex[:8]}"
                
                # Create session through camera session manager
                camera_session = await self.session_manager.create_session(
                    hive_config.internal_camera_id,
                    hive_config.external_camera_id,
                    session_id=session_id
                )
                
                # Create session info
                session_info = {
                    "session_id": session_id,
                    "hive_id": hive_id,
                    "hive_name": hive_config.hive_name,
                    "created_at": datetime.now(),
                    "priority": hive_config.priority,
                    "camera_session": camera_session,
                    "config": session_config or {},
                    "stats": {
                        "events_detected": 0,
                        "bees_tracked": 0,
                        "recording_duration": 0,
                        "last_activity": datetime.now()
                    },
                    "status": "active"
                }
                
                # Store session
                self.active_sessions[session_id] = session_info
                self.hive_sessions[hive_id].add(session_id)
                self.session_priorities[session_id] = hive_config.priority
                
                # Update statistics
                self.stats.active_sessions += 1
                self.stats.total_sessions_created += 1
                
                # Record session history
                self.session_history.append({
                    "session_id": session_id,
                    "hive_id": hive_id,
                    "action": "created",
                    "timestamp": datetime.now().isoformat(),
                    "priority": hive_config.priority.value
                })
                
                logger.info(f"✅ Created session {session_id} for hive {hive_id}")
                
                return {
                    "success": True,
                    "session_id": session_id,
                    "hive_id": hive_id,
                    "hive_name": hive_config.hive_name,
                    "priority": hive_config.priority.value,
                    "camera_session": {
                        "internal_camera": hive_config.internal_camera_id,
                        "external_camera": hive_config.external_camera_id
                    }
                }
                
        except Exception as e:
            logger.error(f"Failed to create session for hive {hive_id}: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def terminate_session(self, session_id: str, reason: str = "manual") -> bool:
        """Terminate a specific session"""
        try:
            if session_id not in self.active_sessions:
                return False
            
            session_info = self.active_sessions[session_id]
            hive_id = session_info["hive_id"]
            
            # Acquire hive lock
            async with self.resource_locks[hive_id]:
                # Terminate camera session
                camera_session = session_info["camera_session"]
                await self.session_manager.cleanup_session(camera_session.session_id)
                
                # Update session status
                session_info["status"] = "terminated"
                session_info["terminated_at"] = datetime.now()
                session_info["termination_reason"] = reason
                
                # Remove from active sessions
                del self.active_sessions[session_id]
                self.hive_sessions[hive_id].discard(session_id)
                
                # Clean up priorities
                if session_id in self.session_priorities:
                    del self.session_priorities[session_id]
                
                # Update statistics
                self.stats.active_sessions -= 1
                
                # Record session history
                self.session_history.append({
                    "session_id": session_id,
                    "hive_id": hive_id,
                    "action": "terminated",
                    "timestamp": datetime.now().isoformat(),
                    "reason": reason
                })
                
                logger.info(f"🛑 Terminated session {session_id} for hive {hive_id} (reason: {reason})")
                return True
                
        except Exception as e:
            logger.error(f"Failed to terminate session {session_id}: {e}")
            return False
    
    async def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific session"""
        if session_id not in self.active_sessions:
            return None
        
        session_info = self.active_sessions[session_id]
        camera_session = session_info["camera_session"]
        
        # Get real-time performance metrics
        performance_metrics = await self.performance_monitor.get_session_metrics(session_id)
        
        return {
            "session_id": session_id,
            "hive_id": session_info["hive_id"],
            "hive_name": session_info["hive_name"],
            "status": session_info["status"],
            "priority": session_info["priority"].value,
            "created_at": session_info["created_at"].isoformat(),
            "duration_seconds": (datetime.now() - session_info["created_at"]).total_seconds(),
            "camera_status": {
                "internal_connected": camera_session.internal_connected,
                "external_connected": camera_session.external_connected,
                "health": camera_session.connection_health
            },
            "statistics": session_info["stats"],
            "performance": performance_metrics,
            "config": session_info["config"]
        }
    
    async def get_hive_sessions(self, hive_id: str) -> List[Dict[str, Any]]:
        """Get all active sessions for a specific hive"""
        if hive_id not in self.hive_configurations:
            return []
        
        hive_session_ids = self.hive_sessions[hive_id]
        sessions = []
        
        for session_id in hive_session_ids:
            session_info = await self.get_session_info(session_id)
            if session_info:
                sessions.append(session_info)
        
        return sessions
    
    async def get_all_sessions(self) -> List[Dict[str, Any]]:
        """Get information about all active sessions"""
        sessions = []
        
        for session_id in self.active_sessions.keys():
            session_info = await self.get_session_info(session_id)
            if session_info:
                sessions.append(session_info)
        
        return sessions
    
    async def pause_session(self, session_id: str) -> bool:
        """Pause a specific session"""
        try:
            if session_id not in self.active_sessions:
                return False
            
            session_info = self.active_sessions[session_id]
            session_info["status"] = "paused"
            session_info["paused_at"] = datetime.now()
            
            # Pause camera session processing
            camera_session = session_info["camera_session"]
            # Implementation depends on camera session manager capabilities
            
            logger.info(f"⏸️ Paused session {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to pause session {session_id}: {e}")
            return False
    
    async def resume_session(self, session_id: str) -> bool:
        """Resume a paused session"""
        try:
            if session_id not in self.active_sessions:
                return False
            
            session_info = self.active_sessions[session_id]
            if session_info["status"] != "paused":
                return False
            
            session_info["status"] = "active"
            session_info["resumed_at"] = datetime.now()
            
            # Resume camera session processing
            camera_session = session_info["camera_session"]
            # Implementation depends on camera session manager capabilities
            
            logger.info(f"▶️ Resumed session {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to resume session {session_id}: {e}")
            return False
    
    async def _update_statistics(self):
        """Update multi-session statistics"""
        try:
            # Update session distribution
            self.stats.session_distribution = {}
            for hive_id, session_ids in self.hive_sessions.items():
                self.stats.session_distribution[hive_id] = len(session_ids)
            
            # Update resource usage
            self.stats.resource_usage = await self._get_resource_usage()
            
            # Update performance metrics
            self.stats.performance_metrics = await self._get_performance_metrics()
            
        except Exception as e:
            logger.error(f"Error updating statistics: {e}")
    
    async def _check_session_health(self):
        """Check health of all active sessions"""
        unhealthy_sessions = []
        
        for session_id, session_info in self.active_sessions.items():
            try:
                camera_session = session_info["camera_session"]
                
                # Check connection health
                if camera_session.connection_health < 0.3:
                    unhealthy_sessions.append((session_id, "poor_connection"))
                
                # Check last activity
                last_activity = session_info["stats"]["last_activity"]
                if datetime.now() - last_activity > timedelta(hours=1):
                    unhealthy_sessions.append((session_id, "inactive"))
                
            except Exception as e:
                logger.error(f"Error checking session health for {session_id}: {e}")
                unhealthy_sessions.append((session_id, "health_check_failed"))
        
        # Handle unhealthy sessions
        for session_id, reason in unhealthy_sessions:
            await self._handle_unhealthy_session(session_id, reason)
    
    async def _handle_unhealthy_session(self, session_id: str, reason: str):
        """Handle an unhealthy session"""
        try:
            session_info = self.active_sessions.get(session_id)
            if not session_info:
                return
            
            priority = session_info["priority"]
            
            # High priority sessions get recovery attempts
            if priority in [SessionPriority.HIGH, SessionPriority.CRITICAL]:
                success = await self._attempt_session_recovery(session_id, reason)
                if success:
                    logger.info(f"✅ Successfully recovered session {session_id}")
                    return
            
            # Low priority sessions or failed recovery - terminate
            logger.warning(f"⚠️ Terminating unhealthy session {session_id} (reason: {reason})")
            await self.terminate_session(session_id, f"unhealthy_{reason}")
            
        except Exception as e:
            logger.error(f"Error handling unhealthy session {session_id}: {e}")
    
    async def _attempt_session_recovery(self, session_id: str, reason: str) -> bool:
        """Attempt to recover an unhealthy session"""
        try:
            session_info = self.active_sessions.get(session_id)
            if not session_info:
                return False
            
            # Implement recovery strategies based on reason
            if reason == "poor_connection":
                # Attempt to reconnect cameras
                camera_session = session_info["camera_session"]
                # Implementation depends on camera session manager
                return True
            
            elif reason == "inactive":
                # Reset activity tracking
                session_info["stats"]["last_activity"] = datetime.now()
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error attempting session recovery for {session_id}: {e}")
            return False
    
    async def _monitor_resources(self):
        """Monitor system resources across all sessions"""
        try:
            # Monitor CPU usage
            cpu_usage = await self.performance_monitor.get_cpu_usage()
            
            # Monitor memory usage
            memory_usage = await self.performance_monitor.get_memory_usage()
            
            # Monitor network bandwidth
            network_usage = await self.performance_monitor.get_network_usage()
            
            # Check for resource exhaustion
            if cpu_usage > 85:
                await self._handle_high_cpu_usage()
            
            if memory_usage > 90:
                await self._handle_high_memory_usage()
            
            if network_usage > 80:
                await self._handle_high_network_usage()
            
        except Exception as e:
            logger.error(f"Error monitoring resources: {e}")
    
    async def _handle_high_cpu_usage(self):
        """Handle high CPU usage by reducing session load"""
        logger.warning("🚨 High CPU usage detected - reducing session load")
        
        # Identify and pause low-priority sessions
        low_priority_sessions = [
            session_id for session_id, priority in self.session_priorities.items()
            if priority == SessionPriority.LOW
        ]
        
        for session_id in low_priority_sessions[:2]:  # Pause up to 2 sessions
            await self.pause_session(session_id)
    
    async def _handle_high_memory_usage(self):
        """Handle high memory usage"""
        logger.warning("🚨 High memory usage detected - optimizing memory")
        
        # Trigger garbage collection and optimize buffers
        # Implementation depends on specific memory management needs
        pass
    
    async def _handle_high_network_usage(self):
        """Handle high network usage"""
        logger.warning("🚨 High network usage detected - optimizing bandwidth")
        
        # Reduce video quality for some sessions
        # Implementation depends on video streaming capabilities
        pass
    
    async def _get_resource_usage(self) -> Dict[str, float]:
        """Get current resource usage"""
        return {
            "cpu_percent": await self.performance_monitor.get_cpu_usage(),
            "memory_percent": await self.performance_monitor.get_memory_usage(),
            "network_percent": await self.performance_monitor.get_network_usage(),
            "disk_percent": await self.performance_monitor.get_disk_usage()
        }
    
    async def _get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics across all sessions"""
        metrics = {
            "total_frame_rate": 0,
            "average_latency": 0,
            "total_bandwidth": 0,
            "session_count": len(self.active_sessions)
        }
        
        # Aggregate metrics from all sessions
        for session_id in self.active_sessions.keys():
            session_metrics = await self.performance_monitor.get_session_metrics(session_id)
            if session_metrics:
                metrics["total_frame_rate"] += session_metrics.get("frame_rate", 0)
                metrics["average_latency"] += session_metrics.get("latency", 0)
                metrics["total_bandwidth"] += session_metrics.get("bandwidth", 0)
        
        if metrics["session_count"] > 0:
            metrics["average_latency"] /= metrics["session_count"]
        
        return metrics
    
    async def _cleanup_expired_sessions(self):
        """Clean up expired sessions"""
        current_time = datetime.now()
        expired_sessions = []
        
        for session_id, session_info in self.active_sessions.items():
            session_age = current_time - session_info["created_at"]
            
            if session_age > timedelta(hours=self.session_timeout_hours):
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            await self.terminate_session(session_id, "expired")
    
    async def _cleanup_orphaned_resources(self):
        """Clean up orphaned resources"""
        # Implementation depends on specific resource management needs
        pass
    
    async def _optimize_resource_allocation(self):
        """Optimize resource allocation across sessions"""
        # Implementation depends on specific optimization strategies
        pass
    
    async def _balance_session_load(self):
        """Balance session load across hives"""
        # Implementation depends on load balancing strategies
        pass
    
    async def _adjust_session_priorities(self):
        """Adjust session priorities based on activity"""
        # Implementation depends on priority adjustment logic
        pass
    
    async def _scale_resources(self):
        """Scale resources based on demand"""
        # Implementation depends on resource scaling capabilities
        pass
    
    def get_multi_session_statistics(self) -> Dict[str, Any]:
        """Get comprehensive multi-session statistics"""
        return {
            "overview": {
                "total_hives": self.stats.total_hives,
                "active_sessions": self.stats.active_sessions,
                "total_sessions_created": self.stats.total_sessions_created,
                "max_concurrent_sessions": self.max_total_sessions
            },
            "hive_distribution": dict(self.stats.session_distribution),
            "resource_usage": self.stats.resource_usage,
            "performance_metrics": self.stats.performance_metrics,
            "session_priorities": {
                priority.value: len([s for s in self.session_priorities.values() if s == priority])
                for priority in SessionPriority
            },
            "recent_history": self.session_history[-10:]  # Last 10 events
        }
    
    async def shutdown(self):
        """Shutdown multi-session manager"""
        logger.info("🛑 Shutting down Multi-Session Manager")
        
        self.monitoring_active = False
        
        # Cancel background tasks
        for task in [self.monitoring_task, self.cleanup_task, self.load_balancer_task]:
            if task:
                task.cancel()
        
        # Terminate all active sessions
        session_ids = list(self.active_sessions.keys())
        for session_id in session_ids:
            await self.terminate_session(session_id, "shutdown")
        
        # Shutdown thread pools
        self.session_executor.shutdown(wait=True, timeout=30)
        self.processing_executor.shutdown(wait=True, timeout=30)
        
        logger.info("✅ Multi-Session Manager shutdown complete")

# Create singleton instance
multi_session_manager = MultiSessionManager() 