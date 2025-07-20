"""
Comprehensive Error Handler Service - Advanced error handling and retry mechanisms
"""
import asyncio
import logging
import time
from typing import Dict, Optional, Callable, Any, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import traceback
from enum import Enum

logger = logging.getLogger(__name__)

class ErrorType(Enum):
    CAMERA_DISCONNECTION = "camera_disconnection"
    NETWORK_FAILURE = "network_failure"
    WEBSOCKET_ERROR = "websocket_error"
    VIDEO_PROCESSING_ERROR = "video_processing_error"
    SESSION_ERROR = "session_error"
    STORAGE_ERROR = "storage_error"
    EMAIL_ERROR = "email_error"
    GENERAL_ERROR = "general_error"

class ErrorSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class ErrorInfo:
    error_type: ErrorType
    severity: ErrorSeverity
    message: str
    timestamp: datetime
    session_id: Optional[str] = None
    camera_type: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    last_retry: Optional[datetime] = None
    resolved: bool = False
    stack_trace: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class RetryConfig:
    """Configuration for retry mechanisms"""
    
    def __init__(self):
        self.base_delay = 1.0  # Base delay in seconds
        self.max_delay = 60.0  # Maximum delay in seconds
        self.exponential_base = 2.0  # Base for exponential backoff
        self.jitter = True  # Add random jitter to avoid thundering herd
        
        # Retry limits per error type
        self.retry_limits = {
            ErrorType.CAMERA_DISCONNECTION: 5,
            ErrorType.NETWORK_FAILURE: 3,
            ErrorType.WEBSOCKET_ERROR: 3,
            ErrorType.VIDEO_PROCESSING_ERROR: 2,
            ErrorType.SESSION_ERROR: 3,
            ErrorType.STORAGE_ERROR: 2,
            ErrorType.EMAIL_ERROR: 3,
            ErrorType.GENERAL_ERROR: 2
        }
        
        # Cooldown periods (seconds) before allowing new attempts
        self.cooldown_periods = {
            ErrorType.CAMERA_DISCONNECTION: 10,
            ErrorType.NETWORK_FAILURE: 5,
            ErrorType.WEBSOCKET_ERROR: 5,
            ErrorType.VIDEO_PROCESSING_ERROR: 15,
            ErrorType.SESSION_ERROR: 10,
            ErrorType.STORAGE_ERROR: 30,
            ErrorType.EMAIL_ERROR: 60,
            ErrorType.GENERAL_ERROR: 10
        }

class ErrorHandler:
    """Comprehensive error handling with intelligent retry mechanisms"""
    
    def __init__(self):
        self.retry_config = RetryConfig()
        self.active_errors: Dict[str, ErrorInfo] = {}
        self.error_history: List[ErrorInfo] = []
        self.max_history_size = 1000
        
        # Circuit breaker states
        self.circuit_breakers: Dict[str, Dict[str, Any]] = {}
        
        # Recovery callbacks
        self.recovery_callbacks: Dict[ErrorType, List[Callable]] = {
            error_type: [] for error_type in ErrorType
        }
        
        # Error metrics
        self.error_metrics = {
            "total_errors": 0,
            "errors_by_type": {error_type.value: 0 for error_type in ErrorType},
            "errors_by_severity": {severity.value: 0 for severity in ErrorSeverity},
            "successful_recoveries": 0,
            "failed_recoveries": 0
        }
        
        logger.info("🛡️ Error Handler initialized with comprehensive retry mechanisms")
    
    def register_recovery_callback(self, error_type: ErrorType, callback: Callable):
        """Register a recovery callback for specific error types"""
        self.recovery_callbacks[error_type].append(callback)
        logger.info(f"Recovery callback registered for {error_type.value}")
    
    async def handle_error(self, 
                          error_type: ErrorType, 
                          error_message: str,
                          severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                          session_id: Optional[str] = None,
                          camera_type: Optional[str] = None,
                          exception: Optional[Exception] = None,
                          metadata: Dict[str, Any] = None) -> str:
        """
        Handle an error with automatic retry and recovery
        
        Returns:
            str: Error ID for tracking
        """
        error_id = f"{error_type.value}_{int(time.time() * 1000)}"
        
        # Create error info
        error_info = ErrorInfo(
            error_type=error_type,
            severity=severity,
            message=error_message,
            timestamp=datetime.now(),
            session_id=session_id,
            camera_type=camera_type,
            max_retries=self.retry_config.retry_limits[error_type],
            stack_trace=traceback.format_exc() if exception else None,
            metadata=metadata or {}
        )
        
        # Store error
        self.active_errors[error_id] = error_info
        self.error_history.append(error_info)
        
        # Update metrics
        self.error_metrics["total_errors"] += 1
        self.error_metrics["errors_by_type"][error_type.value] += 1
        self.error_metrics["errors_by_severity"][severity.value] += 1
        
        # Log error
        log_level = {
            ErrorSeverity.LOW: logging.INFO,
            ErrorSeverity.MEDIUM: logging.WARNING,
            ErrorSeverity.HIGH: logging.ERROR,
            ErrorSeverity.CRITICAL: logging.CRITICAL
        }[severity]
        
        logger.log(log_level, f"🚨 Error {error_id}: {error_message}")
        if session_id:
            logger.log(log_level, f"   Session: {session_id}")
        if camera_type:
            logger.log(log_level, f"   Camera: {camera_type}")
        
        # Check circuit breaker
        if self._is_circuit_breaker_open(error_type, session_id):
            logger.warning(f"Circuit breaker OPEN for {error_type.value} - skipping retry")
            return error_id
        
        # Attempt recovery
        await self._attempt_recovery(error_id, error_info)
        
        return error_id
    
    async def _attempt_recovery(self, error_id: str, error_info: ErrorInfo):
        """Attempt to recover from an error with retry logic"""
        if error_info.retry_count >= error_info.max_retries:
            logger.error(f"Max retries exceeded for error {error_id}")
            self._update_circuit_breaker(error_info.error_type, error_info.session_id, success=False)
            return
        
        # Calculate delay with exponential backoff
        delay = min(
            self.retry_config.base_delay * (self.retry_config.exponential_base ** error_info.retry_count),
            self.retry_config.max_delay
        )
        
        # Add jitter to prevent thundering herd
        if self.retry_config.jitter:
            import random
            delay *= (0.5 + random.random() * 0.5)
        
        logger.info(f"Retrying error {error_id} in {delay:.2f} seconds (attempt {error_info.retry_count + 1}/{error_info.max_retries})")
        
        # Wait before retry
        await asyncio.sleep(delay)
        
        # Update retry info
        error_info.retry_count += 1
        error_info.last_retry = datetime.now()
        
        # Execute recovery callbacks
        recovery_success = False
        for callback in self.recovery_callbacks[error_info.error_type]:
            try:
                if asyncio.iscoroutinefunction(callback):
                    result = await callback(error_info)
                else:
                    result = callback(error_info)
                
                if result:
                    recovery_success = True
                    break
                    
            except Exception as e:
                logger.error(f"Recovery callback failed for {error_id}: {e}")
                continue
        
        if recovery_success:
            logger.info(f"✅ Recovery successful for error {error_id}")
            error_info.resolved = True
            self.error_metrics["successful_recoveries"] += 1
            self._update_circuit_breaker(error_info.error_type, error_info.session_id, success=True)
            
            # Remove from active errors
            if error_id in self.active_errors:
                del self.active_errors[error_id]
        else:
            logger.warning(f"❌ Recovery failed for error {error_id}")
            self.error_metrics["failed_recoveries"] += 1
            
            # Schedule next retry if within limits
            if error_info.retry_count < error_info.max_retries:
                asyncio.create_task(self._attempt_recovery(error_id, error_info))
            else:
                self._update_circuit_breaker(error_info.error_type, error_info.session_id, success=False)
    
    def _is_circuit_breaker_open(self, error_type: ErrorType, session_id: Optional[str]) -> bool:
        """Check if circuit breaker is open for this error type/session"""
        breaker_key = f"{error_type.value}_{session_id or 'global'}"
        
        if breaker_key not in self.circuit_breakers:
            return False
        
        breaker = self.circuit_breakers[breaker_key]
        
        # Check if cooldown period has passed
        if datetime.now() - breaker['last_failure'] > timedelta(seconds=self.retry_config.cooldown_periods[error_type]):
            # Reset circuit breaker
            del self.circuit_breakers[breaker_key]
            return False
        
        return breaker['is_open']
    
    def _update_circuit_breaker(self, error_type: ErrorType, session_id: Optional[str], success: bool):
        """Update circuit breaker state"""
        breaker_key = f"{error_type.value}_{session_id or 'global'}"
        
        if success:
            # Success - reset circuit breaker
            if breaker_key in self.circuit_breakers:
                del self.circuit_breakers[breaker_key]
        else:
            # Failure - open circuit breaker
            self.circuit_breakers[breaker_key] = {
                'is_open': True,
                'last_failure': datetime.now(),
                'failure_count': self.circuit_breakers.get(breaker_key, {}).get('failure_count', 0) + 1
            }
    
    def get_error_status(self, error_id: str) -> Optional[ErrorInfo]:
        """Get status of a specific error"""
        return self.active_errors.get(error_id)
    
    def get_active_errors(self) -> Dict[str, ErrorInfo]:
        """Get all active errors"""
        return self.active_errors.copy()
    
    def get_error_metrics(self) -> Dict[str, Any]:
        """Get error metrics and statistics"""
        return {
            **self.error_metrics,
            "active_errors_count": len(self.active_errors),
            "circuit_breakers_open": len(self.circuit_breakers),
            "error_history_size": len(self.error_history)
        }
    
    def get_error_history(self, limit: int = 100) -> List[ErrorInfo]:
        """Get recent error history"""
        return self.error_history[-limit:]
    
    def clear_resolved_errors(self):
        """Clear resolved errors from active list"""
        resolved_count = 0
        for error_id, error_info in list(self.active_errors.items()):
            if error_info.resolved:
                del self.active_errors[error_id]
                resolved_count += 1
        
        logger.info(f"Cleared {resolved_count} resolved errors")
    
    def cleanup_old_history(self):
        """Clean up old error history"""
        if len(self.error_history) > self.max_history_size:
            removed = len(self.error_history) - self.max_history_size
            self.error_history = self.error_history[-self.max_history_size:]
            logger.info(f"Cleaned up {removed} old error history entries")

# Create singleton instance
error_handler = ErrorHandler() 