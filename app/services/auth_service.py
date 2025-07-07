"""
Authentication and Authorization Service
Comprehensive user management with JWT tokens, RBAC, and security features
"""
import jwt
import bcrypt
import asyncio
import logging
import time
import secrets
from typing import Dict, List, Optional, Any, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import sqlite3
import json
from pathlib import Path

logger = logging.getLogger(__name__)

class Permission(Enum):
    """System permissions"""
    # Video operations
    VIEW_VIDEOS = "view_videos"
    RECORD_VIDEOS = "record_videos"
    DELETE_VIDEOS = "delete_videos"
    DOWNLOAD_VIDEOS = "download_videos"
    
    # Session management
    CREATE_SESSIONS = "create_sessions"
    MANAGE_SESSIONS = "manage_sessions"
    VIEW_SESSIONS = "view_sessions"
    
    # Configuration
    VIEW_CONFIG = "view_config"
    EDIT_CONFIG = "edit_config"
    RESET_CONFIG = "reset_config"
    
    # User management
    VIEW_USERS = "view_users"
    CREATE_USERS = "create_users"
    EDIT_USERS = "edit_users"
    DELETE_USERS = "delete_users"
    
    # System administration
    VIEW_SYSTEM_STATUS = "view_system_status"
    MANAGE_BACKUPS = "manage_backups"
    VIEW_LOGS = "view_logs"
    SYSTEM_ADMIN = "system_admin"
    
    # Monitoring
    VIEW_METRICS = "view_metrics"
    EXPORT_DATA = "export_data"

class Role(Enum):
    """User roles with associated permissions"""
    ADMIN = "admin"
    OPERATOR = "operator"
    VIEWER = "viewer"
    GUEST = "guest"

@dataclass
class User:
    """User account information"""
    user_id: str
    username: str
    email: str
    full_name: str
    role: Role
    permissions: Set[Permission]
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    last_login: Optional[datetime] = None
    password_hash: Optional[str] = None
    failed_login_attempts: int = 0
    locked_until: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Session:
    """User session information"""
    session_id: str
    user_id: str
    username: str
    role: Role
    permissions: Set[Permission]
    created_at: datetime
    last_activity: datetime
    expires_at: datetime
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    is_active: bool = True

@dataclass
class LoginAttempt:
    """Login attempt tracking"""
    attempt_id: str
    username: str
    ip_address: str
    timestamp: datetime
    success: bool
    failure_reason: Optional[str] = None
    user_agent: Optional[str] = None

class AuthService:
    """Comprehensive authentication and authorization service"""
    
    def __init__(self, secret_key: Optional[str] = None, db_path: str = "/data/auth.db"):
        self.secret_key = secret_key or secrets.token_urlsafe(32)
        self.db_path = Path(db_path)
        
        # Create database directory
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        # Configuration
        self.jwt_algorithm = "HS256"
        self.token_expiry_hours = 24
        self.refresh_token_expiry_days = 30
        self.max_failed_attempts = 5
        self.lockout_duration_minutes = 30
        self.session_timeout_minutes = 120
        self.password_min_length = 8
        self.require_password_complexity = True
        
        # Runtime state
        self.active_sessions: Dict[str, Session] = {}
        self.blacklisted_tokens: Set[str] = set()
        
        # Role-Permission mapping
        self.role_permissions = self._define_role_permissions()
        
        # Security monitoring
        self.security_events: List[Dict[str, Any]] = []
        self.max_security_events = 1000
        
        # Background tasks
        self.cleanup_task = None
        self.monitoring_active = False
        
        logger.info("🔐 Authentication Service initialized")
        
        # Create default admin user if none exists
        self._create_default_admin()
        
        # Start background tasks
        self._start_background_tasks()
    
    def _init_database(self):
        """Initialize SQLite database for user management"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                        user_id TEXT PRIMARY KEY,
                        username TEXT UNIQUE NOT NULL,
                        email TEXT UNIQUE NOT NULL,
                        full_name TEXT NOT NULL,
                        role TEXT NOT NULL,
                        password_hash TEXT NOT NULL,
                        is_active BOOLEAN DEFAULT TRUE,
                        created_at TEXT NOT NULL,
                        last_login TEXT,
                        failed_login_attempts INTEGER DEFAULT 0,
                        locked_until TEXT,
                        metadata TEXT DEFAULT '{}'
                    )
                """)
                
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS sessions (
                        session_id TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        username TEXT NOT NULL,
                        role TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        last_activity TEXT NOT NULL,
                        expires_at TEXT NOT NULL,
                        ip_address TEXT,
                        user_agent TEXT,
                        is_active BOOLEAN DEFAULT TRUE,
                        FOREIGN KEY (user_id) REFERENCES users (user_id)
                    )
                """)
                
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS login_attempts (
                        attempt_id TEXT PRIMARY KEY,
                        username TEXT NOT NULL,
                        ip_address TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        success BOOLEAN NOT NULL,
                        failure_reason TEXT,
                        user_agent TEXT
                    )
                """)
                
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS security_events (
                        event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        event_type TEXT NOT NULL,
                        user_id TEXT,
                        username TEXT,
                        ip_address TEXT,
                        timestamp TEXT NOT NULL,
                        details TEXT,
                        severity TEXT DEFAULT 'INFO'
                    )
                """)
                
                conn.commit()
                logger.info("📚 Authentication database initialized")
        except Exception as e:
            logger.error(f"Failed to initialize auth database: {e}")
    
    def _define_role_permissions(self) -> Dict[Role, Set[Permission]]:
        """Define permissions for each role"""
        return {
            Role.ADMIN: {
                # Full access to everything
                Permission.VIEW_VIDEOS, Permission.RECORD_VIDEOS, Permission.DELETE_VIDEOS, Permission.DOWNLOAD_VIDEOS,
                Permission.CREATE_SESSIONS, Permission.MANAGE_SESSIONS, Permission.VIEW_SESSIONS,
                Permission.VIEW_CONFIG, Permission.EDIT_CONFIG, Permission.RESET_CONFIG,
                Permission.VIEW_USERS, Permission.CREATE_USERS, Permission.EDIT_USERS, Permission.DELETE_USERS,
                Permission.VIEW_SYSTEM_STATUS, Permission.MANAGE_BACKUPS, Permission.VIEW_LOGS, Permission.SYSTEM_ADMIN,
                Permission.VIEW_METRICS, Permission.EXPORT_DATA
            },
            Role.OPERATOR: {
                # Operational access
                Permission.VIEW_VIDEOS, Permission.RECORD_VIDEOS, Permission.DOWNLOAD_VIDEOS,
                Permission.CREATE_SESSIONS, Permission.MANAGE_SESSIONS, Permission.VIEW_SESSIONS,
                Permission.VIEW_CONFIG, Permission.EDIT_CONFIG,
                Permission.VIEW_SYSTEM_STATUS, Permission.VIEW_METRICS, Permission.EXPORT_DATA
            },
            Role.VIEWER: {
                # Read-only access
                Permission.VIEW_VIDEOS, Permission.DOWNLOAD_VIDEOS,
                Permission.VIEW_SESSIONS, Permission.VIEW_CONFIG,
                Permission.VIEW_SYSTEM_STATUS, Permission.VIEW_METRICS
            },
            Role.GUEST: {
                # Minimal access
                Permission.VIEW_VIDEOS, Permission.VIEW_SESSIONS, Permission.VIEW_SYSTEM_STATUS
            }
        }
    
    def _create_default_admin(self):
        """Create default admin user if none exists"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM users WHERE role = ?", (Role.ADMIN.value,))
                admin_count = cursor.fetchone()[0]
                
                if admin_count == 0:
                    # Create default admin user
                    admin_user = User(
                        user_id="admin_001",
                        username="admin",
                        email="admin@queentrack.local",
                        full_name="System Administrator",
                        role=Role.ADMIN,
                        permissions=self.role_permissions[Role.ADMIN]
                    )
                    
                    # Default password: "admin123" (should be changed on first login)
                    password_hash = self._hash_password("admin123")
                    
                    conn.execute("""
                        INSERT INTO users (user_id, username, email, full_name, role, password_hash, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        admin_user.user_id, admin_user.username, admin_user.email,
                        admin_user.full_name, admin_user.role.value, password_hash,
                        admin_user.created_at.isoformat()
                    ))
                    
                    conn.commit()
                    logger.info("👨‍💼 Default admin user created (username: admin, password: admin123)")
                    
        except Exception as e:
            logger.error(f"Error creating default admin: {e}")
    
    def _start_background_tasks(self):
        """Start background authentication tasks"""
        self.monitoring_active = True
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("🚀 Authentication background tasks started")
    
    async def _cleanup_loop(self):
        """Background cleanup of expired sessions and tokens"""
        while self.monitoring_active:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                await self._cleanup_expired_sessions()
                await self._cleanup_old_events()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in auth cleanup loop: {e}")
    
    async def _cleanup_expired_sessions(self):
        """Remove expired sessions"""
        try:
            current_time = datetime.now()
            expired_sessions = []
            
            # Check in-memory sessions
            for session_id, session in self.active_sessions.items():
                if current_time > session.expires_at:
                    expired_sessions.append(session_id)
            
            # Remove expired sessions
            for session_id in expired_sessions:
                del self.active_sessions[session_id]
            
            # Clean up database
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    UPDATE sessions SET is_active = FALSE 
                    WHERE expires_at < ? AND is_active = TRUE
                """, (current_time.isoformat(),))
                conn.commit()
            
            if expired_sessions:
                logger.info(f"🧹 Cleaned up {len(expired_sessions)} expired sessions")
                
        except Exception as e:
            logger.error(f"Error cleaning up expired sessions: {e}")
    
    async def _cleanup_old_events(self):
        """Clean up old security events"""
        try:
            # Keep only the most recent events
            if len(self.security_events) > self.max_security_events:
                self.security_events = self.security_events[-self.max_security_events:]
            
            # Clean up old database records (older than 90 days)
            cutoff_date = datetime.now() - timedelta(days=90)
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    DELETE FROM login_attempts WHERE timestamp < ?
                """, (cutoff_date.isoformat(),))
                
                conn.execute("""
                    DELETE FROM security_events WHERE timestamp < ?
                """, (cutoff_date.isoformat(),))
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error cleaning up old events: {e}")
    
    def _hash_password(self, password: str) -> str:
        """Hash password using bcrypt"""
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')
    
    def _verify_password(self, password: str, password_hash: str) -> bool:
        """Verify password against hash"""
        return bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8'))
    
    def _validate_password_complexity(self, password: str) -> bool:
        """Validate password complexity requirements"""
        if len(password) < self.password_min_length:
            return False
        
        if self.require_password_complexity:
            has_upper = any(c.isupper() for c in password)
            has_lower = any(c.islower() for c in password)
            has_digit = any(c.isdigit() for c in password)
            has_special = any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password)
            
            return has_upper and has_lower and has_digit and has_special
        
        return True
    
    def _generate_jwt_token(self, user: User, session_id: str) -> str:
        """Generate JWT token for user"""
        payload = {
            "user_id": user.user_id,
            "username": user.username,
            "role": user.role.value,
            "session_id": session_id,
            "permissions": [p.value for p in user.permissions],
            "iat": int(time.time()),
            "exp": int(time.time()) + (self.token_expiry_hours * 3600)
        }
        
        return jwt.encode(payload, self.secret_key, algorithm=self.jwt_algorithm)
    
    def _decode_jwt_token(self, token: str) -> Dict[str, Any]:
        """Decode and validate JWT token"""
        try:
            # Check if token is blacklisted
            if token in self.blacklisted_tokens:
                raise jwt.InvalidTokenError("Token is blacklisted")
            
            payload = jwt.decode(token, self.secret_key, algorithms=[self.jwt_algorithm])
            
            # Verify session is still active
            session_id = payload.get("session_id")
            if session_id not in self.active_sessions:
                raise jwt.InvalidTokenError("Session not found")
            
            session = self.active_sessions[session_id]
            if not session.is_active or datetime.now() > session.expires_at:
                raise jwt.InvalidTokenError("Session expired")
            
            return payload
            
        except jwt.ExpiredSignatureError:
            raise jwt.InvalidTokenError("Token expired")
        except jwt.InvalidTokenError:
            raise jwt.InvalidTokenError("Invalid token")
    
    def _record_security_event(self, event_type: str, user_id: Optional[str] = None,
                             username: Optional[str] = None, ip_address: Optional[str] = None,
                             details: Optional[str] = None, severity: str = "INFO"):
        """Record security event"""
        event = {
            "event_type": event_type,
            "user_id": user_id,
            "username": username,
            "ip_address": ip_address,
            "timestamp": datetime.now().isoformat(),
            "details": details,
            "severity": severity
        }
        
        self.security_events.append(event)
        
        # Save to database
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO security_events 
                    (event_type, user_id, username, ip_address, timestamp, details, severity)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (event_type, user_id, username, ip_address, event["timestamp"], details, severity))
                conn.commit()
        except Exception as e:
            logger.error(f"Error recording security event: {e}")
    
    async def authenticate_user(self, username: str, password: str, 
                              ip_address: Optional[str] = None,
                              user_agent: Optional[str] = None) -> Dict[str, Any]:
        """Authenticate user and create session"""
        attempt_id = f"attempt_{int(time.time() * 1000)}"
        
        try:
            # Get user from database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT user_id, username, email, full_name, role, password_hash, 
                           is_active, failed_login_attempts, locked_until
                    FROM users WHERE username = ?
                """, (username,))
                
                user_data = cursor.fetchone()
            
            if not user_data:
                # Record failed attempt
                self._record_login_attempt(attempt_id, username, ip_address, False, 
                                         "User not found", user_agent)
                self._record_security_event("LOGIN_FAILED", username=username, 
                                          ip_address=ip_address, details="User not found")
                
                return {"success": False, "message": "Invalid credentials"}
            
            user_id, db_username, email, full_name, role_str, password_hash, is_active, failed_attempts, locked_until = user_data
            
            # Check if account is locked
            if locked_until:
                lockout_time = datetime.fromisoformat(locked_until)
                if datetime.now() < lockout_time:
                    self._record_login_attempt(attempt_id, username, ip_address, False, 
                                             "Account locked", user_agent)
                    return {"success": False, "message": "Account is locked"}
            
            # Check if account is active
            if not is_active:
                self._record_login_attempt(attempt_id, username, ip_address, False, 
                                         "Account disabled", user_agent)
                return {"success": False, "message": "Account is disabled"}
            
            # Verify password
            if not self._verify_password(password, password_hash):
                # Increment failed attempts
                failed_attempts += 1
                
                # Lock account if too many failed attempts
                if failed_attempts >= self.max_failed_attempts:
                    lockout_time = datetime.now() + timedelta(minutes=self.lockout_duration_minutes)
                    with sqlite3.connect(self.db_path) as conn:
                        conn.execute("""
                            UPDATE users SET failed_login_attempts = ?, locked_until = ?
                            WHERE user_id = ?
                        """, (failed_attempts, lockout_time.isoformat(), user_id))
                        conn.commit()
                    
                    self._record_security_event("ACCOUNT_LOCKED", user_id=user_id, 
                                              username=username, ip_address=ip_address, 
                                              severity="WARNING")
                    
                    failure_reason = "Account locked due to too many failed attempts"
                else:
                    with sqlite3.connect(self.db_path) as conn:
                        conn.execute("""
                            UPDATE users SET failed_login_attempts = ?
                            WHERE user_id = ?
                        """, (failed_attempts, user_id))
                        conn.commit()
                    
                    failure_reason = "Invalid password"
                
                self._record_login_attempt(attempt_id, username, ip_address, False, 
                                         failure_reason, user_agent)
                self._record_security_event("LOGIN_FAILED", user_id=user_id, 
                                          username=username, ip_address=ip_address, 
                                          details=failure_reason)
                
                return {"success": False, "message": "Invalid credentials"}
            
            # Authentication successful
            role = Role(role_str)
            permissions = self.role_permissions[role]
            
            # Create user object
            user = User(
                user_id=user_id,
                username=db_username,
                email=email,
                full_name=full_name,
                role=role,
                permissions=permissions,
                is_active=is_active
            )
            
            # Create session
            session_id = f"session_{user_id}_{int(time.time() * 1000)}"
            session = Session(
                session_id=session_id,
                user_id=user_id,
                username=db_username,
                role=role,
                permissions=permissions,
                created_at=datetime.now(),
                last_activity=datetime.now(),
                expires_at=datetime.now() + timedelta(minutes=self.session_timeout_minutes),
                ip_address=ip_address,
                user_agent=user_agent
            )
            
            # Store session
            self.active_sessions[session_id] = session
            
            # Save session to database
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO sessions 
                    (session_id, user_id, username, role, created_at, last_activity, 
                     expires_at, ip_address, user_agent)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    session_id, user_id, db_username, role.value,
                    session.created_at.isoformat(), session.last_activity.isoformat(),
                    session.expires_at.isoformat(), ip_address, user_agent
                ))
                
                # Reset failed attempts and clear lockout
                conn.execute("""
                    UPDATE users SET failed_login_attempts = 0, locked_until = NULL, 
                                   last_login = ? WHERE user_id = ?
                """, (datetime.now().isoformat(), user_id))
                
                conn.commit()
            
            # Generate JWT token
            token = self._generate_jwt_token(user, session_id)
            
            # Record successful login
            self._record_login_attempt(attempt_id, username, ip_address, True, None, user_agent)
            self._record_security_event("LOGIN_SUCCESS", user_id=user_id, 
                                      username=username, ip_address=ip_address)
            
            logger.info(f"🔓 User authenticated: {username} (role: {role.value})")
            
            return {
                "success": True,
                "user": {
                    "user_id": user.user_id,
                    "username": user.username,
                    "email": user.email,
                    "full_name": user.full_name,
                    "role": user.role.value,
                    "permissions": [p.value for p in user.permissions]
                },
                "token": token,
                "session_id": session_id,
                "expires_at": session.expires_at.isoformat()
            }
            
        except Exception as e:
            self._record_login_attempt(attempt_id, username, ip_address, False, 
                                     f"Authentication error: {str(e)}", user_agent)
            self._record_security_event("LOGIN_ERROR", username=username, 
                                      ip_address=ip_address, details=str(e), severity="ERROR")
            
            logger.error(f"Authentication error for {username}: {e}")
            return {"success": False, "message": "Authentication failed"}
    
    def _record_login_attempt(self, attempt_id: str, username: str, ip_address: Optional[str],
                            success: bool, failure_reason: Optional[str], user_agent: Optional[str]):
        """Record login attempt"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO login_attempts 
                    (attempt_id, username, ip_address, timestamp, success, failure_reason, user_agent)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    attempt_id, username, ip_address or "unknown", 
                    datetime.now().isoformat(), success, failure_reason, user_agent
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Error recording login attempt: {e}")
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify JWT token and return user information"""
        try:
            payload = self._decode_jwt_token(token)
            
            # Update session activity
            session_id = payload["session_id"]
            if session_id in self.active_sessions:
                self.active_sessions[session_id].last_activity = datetime.now()
            
            return {
                "valid": True,
                "user_id": payload["user_id"],
                "username": payload["username"],
                "role": payload["role"],
                "permissions": payload["permissions"],
                "session_id": session_id
            }
            
        except jwt.InvalidTokenError as e:
            return {"valid": False, "error": str(e)}
    
    def check_permission(self, token: str, required_permission: Permission) -> bool:
        """Check if user has required permission"""
        try:
            payload = self._decode_jwt_token(token)
            user_permissions = [Permission(p) for p in payload["permissions"]]
            return required_permission in user_permissions
        except:
            return False
    
    def logout_user(self, token: str) -> bool:
        """Logout user and invalidate session"""
        try:
            payload = self._decode_jwt_token(token)
            session_id = payload["session_id"]
            
            # Remove session
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]
            
            # Add token to blacklist
            self.blacklisted_tokens.add(token)
            
            # Mark session as inactive in database
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    UPDATE sessions SET is_active = FALSE WHERE session_id = ?
                """, (session_id,))
                conn.commit()
            
            self._record_security_event("LOGOUT", user_id=payload["user_id"], 
                                      username=payload["username"])
            
            logger.info(f"🔒 User logged out: {payload['username']}")
            return True
            
        except Exception as e:
            logger.error(f"Error during logout: {e}")
            return False
    
    def get_auth_statistics(self) -> Dict[str, Any]:
        """Get authentication and security statistics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Get user statistics
                cursor = conn.execute("SELECT COUNT(*) FROM users")
                total_users = cursor.fetchone()[0]
                
                cursor = conn.execute("SELECT COUNT(*) FROM users WHERE is_active = TRUE")
                active_users = cursor.fetchone()[0]
                
                # Get session statistics
                cursor = conn.execute("SELECT COUNT(*) FROM sessions WHERE is_active = TRUE")
                active_sessions = cursor.fetchone()[0]
                
                # Get recent login attempts
                since_24h = (datetime.now() - timedelta(hours=24)).isoformat()
                cursor = conn.execute("""
                    SELECT COUNT(*) FROM login_attempts 
                    WHERE timestamp > ? AND success = TRUE
                """, (since_24h,))
                successful_logins_24h = cursor.fetchone()[0]
                
                cursor = conn.execute("""
                    SELECT COUNT(*) FROM login_attempts 
                    WHERE timestamp > ? AND success = FALSE
                """, (since_24h,))
                failed_logins_24h = cursor.fetchone()[0]
            
            return {
                "users": {
                    "total": total_users,
                    "active": active_users,
                    "locked": total_users - active_users
                },
                "sessions": {
                    "active": active_sessions,
                    "in_memory": len(self.active_sessions)
                },
                "logins_24h": {
                    "successful": successful_logins_24h,
                    "failed": failed_logins_24h,
                    "total": successful_logins_24h + failed_logins_24h
                },
                "security": {
                    "recent_events": len(self.security_events),
                    "blacklisted_tokens": len(self.blacklisted_tokens)
                },
                "configuration": {
                    "token_expiry_hours": self.token_expiry_hours,
                    "max_failed_attempts": self.max_failed_attempts,
                    "lockout_duration_minutes": self.lockout_duration_minutes,
                    "session_timeout_minutes": self.session_timeout_minutes
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting auth statistics: {e}")
            return {"error": str(e)}
    
    async def shutdown(self):
        """Shutdown authentication service"""
        logger.info("🛑 Shutting down Authentication Service")
        
        self.monitoring_active = False
        
        # Cancel background tasks
        if self.cleanup_task:
            self.cleanup_task.cancel()
        
        # Clear active sessions
        self.active_sessions.clear()
        
        logger.info("✅ Authentication Service shutdown complete")

# Create singleton instance
auth_service = AuthService() 