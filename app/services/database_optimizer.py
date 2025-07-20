"""
Database Optimization Service - Advanced database performance and management
"""
import asyncio
import logging
import sqlite3
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import json
import threading
from concurrent.futures import ThreadPoolExecutor
import os

logger = logging.getLogger(__name__)

@dataclass
class IndexInfo:
    """Database index information"""
    index_name: str
    table_name: str
    columns: List[str]
    unique: bool = False
    partial_condition: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    last_used: Optional[datetime] = None
    usage_count: int = 0

@dataclass
class QueryPerformance:
    """Query performance metrics"""
    query_hash: str
    query_text: str
    execution_count: int = 0
    total_time_ms: float = 0
    average_time_ms: float = 0
    min_time_ms: float = float('inf')
    max_time_ms: float = 0
    last_executed: datetime = field(default_factory=datetime.now)
    table_scans: int = 0
    index_usage: List[str] = field(default_factory=list)

@dataclass
class TableStats:
    """Table statistics and metrics"""
    table_name: str
    row_count: int = 0
    size_kb: float = 0
    avg_row_size: float = 0
    last_vacuum: Optional[datetime] = None
    last_analyzed: Optional[datetime] = None
    fragmentation_percent: float = 0
    read_operations: int = 0
    write_operations: int = 0

class DatabaseOptimizer:
    """Comprehensive database optimization and performance monitoring"""
    
    def __init__(self, db_paths: Dict[str, str]):
        """
        Initialize optimizer with database paths
        db_paths: Dictionary mapping database names to file paths
        """
        self.db_paths = db_paths
        self.connections: Dict[str, sqlite3.Connection] = {}
        
        # Performance tracking
        self.query_performance: Dict[str, QueryPerformance] = {}
        self.table_stats: Dict[str, Dict[str, TableStats]] = {}  # db_name -> table_name -> stats
        self.index_registry: Dict[str, Dict[str, IndexInfo]] = {}  # db_name -> index_name -> info
        
        # Configuration
        self.enable_query_monitoring = True
        self.enable_auto_optimization = True
        self.optimization_interval_hours = 6
        self.query_cache_size = 1000
        self.slow_query_threshold_ms = 100
        self.vacuum_threshold_days = 7
        self.analyze_threshold_hours = 24
        
        # Thread pool for database operations
        self.executor = ThreadPoolExecutor(max_workers=3)
        
        # Background tasks
        self.monitoring_task = None
        self.optimization_task = None
        self.monitoring_active = False
        
        # Statistics
        self.optimization_stats = {
            "total_optimizations": 0,
            "indexes_created": 0,
            "vacuum_operations": 0,
            "analyze_operations": 0,
            "slow_queries_optimized": 0,
            "space_reclaimed_mb": 0
        }
        
        logger.info("🗄️ Database Optimizer initialized")
        
        # Initialize connections and start monitoring
        self._initialize_connections()
        self._initialize_monitoring()
        self._start_background_tasks()
    
    def _initialize_connections(self):
        """Initialize database connections with optimization settings"""
        for db_name, db_path in self.db_paths.items():
            try:
                # Ensure database directory exists
                Path(db_path).parent.mkdir(parents=True, exist_ok=True)
                
                # Create connection with optimized settings
                conn = sqlite3.connect(
                    db_path,
                    timeout=30.0,
                    check_same_thread=False,
                    isolation_level=None  # Autocommit mode
                )
                
                # Enable performance optimizations
                conn.execute("PRAGMA journal_mode = WAL")  # Write-Ahead Logging
                conn.execute("PRAGMA synchronous = NORMAL")  # Balanced safety/performance
                conn.execute("PRAGMA cache_size = -64000")  # 64MB cache
                conn.execute("PRAGMA temp_store = MEMORY")  # Use memory for temp tables
                conn.execute("PRAGMA mmap_size = 268435456")  # 256MB memory-mapped I/O
                conn.execute("PRAGMA optimize")  # Enable query optimizer
                
                # Enable foreign key constraints
                conn.execute("PRAGMA foreign_keys = ON")
                
                # Set up query monitoring if enabled
                if self.enable_query_monitoring:
                    self._setup_query_monitoring(conn, db_name)
                
                self.connections[db_name] = conn
                self.table_stats[db_name] = {}
                self.index_registry[db_name] = {}
                
                logger.info(f"✅ Connected to database: {db_name}")
                
            except Exception as e:
                logger.error(f"Failed to connect to database {db_name}: {e}")
    
    def _setup_query_monitoring(self, conn: sqlite3.Connection, db_name: str):
        """Set up query performance monitoring"""
        # This would require custom SQLite extension or query logging
        # For now, we'll implement basic monitoring through wrapper functions
        pass
    
    def _initialize_monitoring(self):
        """Initialize database monitoring and create necessary indexes"""
        for db_name, conn in self.connections.items():
            try:
                # Analyze existing schema
                self._analyze_schema(db_name)
                
                # Create performance monitoring tables
                self._create_monitoring_tables(db_name)
                
                # Create essential indexes
                self._create_essential_indexes(db_name)
                
                # Collect initial statistics
                self._collect_table_statistics(db_name)
                
            except Exception as e:
                logger.error(f"Error initializing monitoring for {db_name}: {e}")
    
    def _create_monitoring_tables(self, db_name: str):
        """Create tables for performance monitoring"""
        conn = self.connections[db_name]
        
        # Query performance tracking table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS query_performance (
                query_hash TEXT PRIMARY KEY,
                query_text TEXT NOT NULL,
                execution_count INTEGER DEFAULT 0,
                total_time_ms REAL DEFAULT 0,
                average_time_ms REAL DEFAULT 0,
                min_time_ms REAL DEFAULT 0,
                max_time_ms REAL DEFAULT 0,
                last_executed TEXT,
                table_scans INTEGER DEFAULT 0,
                index_usage TEXT DEFAULT '[]',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Index usage tracking
        conn.execute("""
            CREATE TABLE IF NOT EXISTS index_usage (
                index_name TEXT NOT NULL,
                table_name TEXT NOT NULL,
                usage_count INTEGER DEFAULT 0,
                last_used TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (index_name, table_name)
            )
        """)
        
        # Table statistics
        conn.execute("""
            CREATE TABLE IF NOT EXISTS table_statistics (
                table_name TEXT PRIMARY KEY,
                row_count INTEGER DEFAULT 0,
                size_kb REAL DEFAULT 0,
                avg_row_size REAL DEFAULT 0,
                last_vacuum TEXT,
                last_analyzed TEXT,
                fragmentation_percent REAL DEFAULT 0,
                read_operations INTEGER DEFAULT 0,
                write_operations INTEGER DEFAULT 0,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
    
    def _create_essential_indexes(self, db_name: str):
        """Create essential indexes for Queen Track system"""
        conn = self.connections[db_name]
        
        essential_indexes = [
            # Session management indexes
            ("idx_sessions_created_at", "sessions", ["created_at"]),
            ("idx_sessions_status", "sessions", ["status"]),
            ("idx_sessions_user_id", "sessions", ["user_id"]),
            
            # Event tracking indexes
            ("idx_events_session_id", "events", ["session_id"]),
            ("idx_events_timestamp", "events", ["timestamp"]),
            ("idx_events_type", "events", ["event_type"]),
            ("idx_events_session_timestamp", "events", ["session_id", "timestamp"]),
            
            # Video recording indexes
            ("idx_videos_session_id", "videos", ["session_id"]),
            ("idx_videos_created_at", "videos", ["created_at"]),
            ("idx_videos_file_path", "videos", ["file_path"]),
            
            # User management indexes
            ("idx_users_username", "users", ["username"], True),  # Unique index
            ("idx_users_email", "users", ["email"], True),  # Unique index
            ("idx_users_active", "users", ["is_active"]),
            
            # Authentication indexes
            ("idx_login_attempts_username", "login_attempts", ["username"]),
            ("idx_login_attempts_timestamp", "login_attempts", ["timestamp"]),
            ("idx_login_attempts_success", "login_attempts", ["success"]),
            
            # Backup tracking indexes
            ("idx_backup_items_created_at", "backup_items", ["created_at"]),
            ("idx_backup_items_type", "backup_items", ["item_type"]),
            ("idx_backup_history_job_id", "backup_history", ["job_id"]),
            
            # Performance monitoring indexes
            ("idx_query_perf_hash", "query_performance", ["query_hash"], True),
            ("idx_query_perf_exec_count", "query_performance", ["execution_count"]),
            ("idx_index_usage_name", "index_usage", ["index_name"])
        ]
        
        for index_info in essential_indexes:
            index_name, table_name, columns = index_info[:3]
            unique = index_info[3] if len(index_info) > 3 else False
            
            self._create_index(db_name, index_name, table_name, columns, unique)
    
    def _create_index(self, db_name: str, index_name: str, table_name: str, 
                     columns: List[str], unique: bool = False, 
                     partial_condition: Optional[str] = None):
        """Create a database index"""
        try:
            conn = self.connections[db_name]
            
            # Check if index already exists
            cursor = conn.execute("""
                SELECT name FROM sqlite_master 
                WHERE type = 'index' AND name = ?
            """, (index_name,))
            
            if cursor.fetchone():
                logger.debug(f"Index {index_name} already exists")
                return
            
            # Build CREATE INDEX statement
            unique_keyword = "UNIQUE " if unique else ""
            columns_str = ", ".join(columns)
            
            sql = f"CREATE {unique_keyword}INDEX {index_name} ON {table_name} ({columns_str})"
            
            if partial_condition:
                sql += f" WHERE {partial_condition}"
            
            # Execute index creation
            start_time = time.time()
            conn.execute(sql)
            conn.commit()
            creation_time = time.time() - start_time
            
            # Register index
            index_info = IndexInfo(
                index_name=index_name,
                table_name=table_name,
                columns=columns,
                unique=unique,
                partial_condition=partial_condition
            )
            
            self.index_registry[db_name][index_name] = index_info
            self.optimization_stats["indexes_created"] += 1
            
            logger.info(f"📊 Created index {index_name} on {table_name}({columns_str}) in {creation_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Failed to create index {index_name}: {e}")
    
    def _analyze_schema(self, db_name: str):
        """Analyze database schema and identify optimization opportunities"""
        try:
            conn = self.connections[db_name]
            
            # Get all tables
            cursor = conn.execute("""
                SELECT name FROM sqlite_master 
                WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
            """)
            
            tables = [row[0] for row in cursor.fetchall()]
            
            for table_name in tables:
                # Analyze table structure
                cursor = conn.execute(f"PRAGMA table_info({table_name})")
                columns = cursor.fetchall()
                
                # Check for foreign key relationships
                cursor = conn.execute(f"PRAGMA foreign_key_list({table_name})")
                foreign_keys = cursor.fetchall()
                
                # Suggest indexes for foreign key columns
                for fk in foreign_keys:
                    from_column = fk[3]  # from column
                    suggested_index = f"idx_{table_name}_{from_column}"
                    
                    if suggested_index not in self.index_registry[db_name]:
                        self._create_index(db_name, suggested_index, table_name, [from_column])
            
        except Exception as e:
            logger.error(f"Error analyzing schema for {db_name}: {e}")
    
    def _collect_table_statistics(self, db_name: str):
        """Collect comprehensive table statistics"""
        try:
            conn = self.connections[db_name]
            
            # Get all user tables
            cursor = conn.execute("""
                SELECT name FROM sqlite_master 
                WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
            """)
            
            tables = [row[0] for row in cursor.fetchall()]
            
            for table_name in tables:
                stats = TableStats(table_name=table_name)
                
                # Get row count
                cursor = conn.execute(f"SELECT COUNT(*) FROM {table_name}")
                stats.row_count = cursor.fetchone()[0]
                
                # Get table size (approximate)
                cursor = conn.execute("""
                    SELECT page_count * page_size as size 
                    FROM pragma_page_count(?), pragma_page_size(?)
                """, (table_name, table_name))
                
                result = cursor.fetchone()
                if result and result[0]:
                    stats.size_kb = result[0] / 1024
                    if stats.row_count > 0:
                        stats.avg_row_size = stats.size_kb * 1024 / stats.row_count
                
                self.table_stats[db_name][table_name] = stats
                
        except Exception as e:
            logger.error(f"Error collecting table statistics for {db_name}: {e}")
    
    def _start_background_tasks(self):
        """Start background optimization tasks"""
        self.monitoring_active = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        self.optimization_task = asyncio.create_task(self._optimization_loop())
        logger.info("🚀 Database optimization background tasks started")
    
    async def _monitoring_loop(self):
        """Background monitoring of database performance"""
        while self.monitoring_active:
            try:
                await asyncio.sleep(300)  # Monitor every 5 minutes
                
                for db_name in self.connections.keys():
                    # Update table statistics
                    await asyncio.get_event_loop().run_in_executor(
                        self.executor, self._collect_table_statistics, db_name
                    )
                    
                    # Check for slow queries
                    await self._analyze_slow_queries(db_name)
                    
                    # Monitor index usage
                    await self._monitor_index_usage(db_name)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in database monitoring loop: {e}")
    
    async def _optimization_loop(self):
        """Background database optimization"""
        while self.monitoring_active:
            try:
                await asyncio.sleep(self.optimization_interval_hours * 3600)
                
                if self.enable_auto_optimization:
                    for db_name in self.connections.keys():
                        await self._optimize_database(db_name)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in database optimization loop: {e}")
    
    async def _analyze_slow_queries(self, db_name: str):
        """Analyze and suggest optimizations for slow queries"""
        try:
            # This would require query logging implementation
            # For now, we'll focus on table scan detection
            
            for query_hash, performance in self.query_performance.items():
                if (performance.average_time_ms > self.slow_query_threshold_ms and 
                    performance.table_scans > 0):
                    
                    await self._suggest_query_optimization(db_name, performance)
            
        except Exception as e:
            logger.error(f"Error analyzing slow queries for {db_name}: {e}")
    
    async def _suggest_query_optimization(self, db_name: str, performance: QueryPerformance):
        """Suggest optimizations for slow queries"""
        try:
            # Analyze query and suggest indexes
            query_text = performance.query_text.lower()
            
            # Look for WHERE clauses that might benefit from indexes
            if "where" in query_text:
                # This is a simplified analysis - in practice, would need SQL parsing
                logger.info(f"💡 Slow query detected: {performance.query_hash[:8]}... "
                           f"(avg: {performance.average_time_ms:.1f}ms)")
            
        except Exception as e:
            logger.error(f"Error suggesting query optimization: {e}")
    
    async def _monitor_index_usage(self, db_name: str):
        """Monitor index usage and identify unused indexes"""
        try:
            conn = self.connections[db_name]
            
            # Check which indexes are being used
            for index_name, index_info in self.index_registry[db_name].items():
                # Update usage statistics
                # This would require SQLite extension or custom monitoring
                pass
            
        except Exception as e:
            logger.error(f"Error monitoring index usage for {db_name}: {e}")
    
    async def _optimize_database(self, db_name: str):
        """Perform comprehensive database optimization"""
        try:
            logger.info(f"🔧 Starting optimization for database: {db_name}")
            
            start_time = time.time()
            
            # Vacuum database if needed
            await self._vacuum_if_needed(db_name)
            
            # Analyze database statistics
            await self._analyze_if_needed(db_name)
            
            # Optimize queries
            await self._optimize_queries(db_name)
            
            # Check for missing indexes
            await self._suggest_missing_indexes(db_name)
            
            # Clean up unused indexes
            await self._cleanup_unused_indexes(db_name)
            
            optimization_time = time.time() - start_time
            self.optimization_stats["total_optimizations"] += 1
            
            logger.info(f"✅ Database optimization completed for {db_name} in {optimization_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Error optimizing database {db_name}: {e}")
    
    async def _vacuum_if_needed(self, db_name: str):
        """Vacuum database if fragmentation is high"""
        try:
            conn = self.connections[db_name]
            
            # Check if vacuum is needed
            should_vacuum = False
            
            for table_name, stats in self.table_stats[db_name].items():
                if (stats.fragmentation_percent > 20 or 
                    (stats.last_vacuum and 
                     datetime.now() - stats.last_vacuum > timedelta(days=self.vacuum_threshold_days))):
                    should_vacuum = True
                    break
            
            if should_vacuum:
                logger.info(f"🧹 Vacuuming database: {db_name}")
                
                start_time = time.time()
                size_before = Path(self.db_paths[db_name]).stat().st_size
                
                conn.execute("VACUUM")
                conn.commit()
                
                size_after = Path(self.db_paths[db_name]).stat().st_size
                space_reclaimed = (size_before - size_after) / 1024 / 1024  # MB
                
                vacuum_time = time.time() - start_time
                self.optimization_stats["vacuum_operations"] += 1
                self.optimization_stats["space_reclaimed_mb"] += space_reclaimed
                
                logger.info(f"✅ Vacuum completed in {vacuum_time:.2f}s, "
                           f"reclaimed {space_reclaimed:.1f}MB")
                
                # Update statistics
                for stats in self.table_stats[db_name].values():
                    stats.last_vacuum = datetime.now()
                    stats.fragmentation_percent = 0
            
        except Exception as e:
            logger.error(f"Error vacuuming database {db_name}: {e}")
    
    async def _analyze_if_needed(self, db_name: str):
        """Run ANALYZE if statistics are stale"""
        try:
            conn = self.connections[db_name]
            
            should_analyze = False
            
            for table_name, stats in self.table_stats[db_name].items():
                if (not stats.last_analyzed or 
                    datetime.now() - stats.last_analyzed > timedelta(hours=self.analyze_threshold_hours)):
                    should_analyze = True
                    break
            
            if should_analyze:
                logger.info(f"📊 Analyzing database: {db_name}")
                
                start_time = time.time()
                conn.execute("ANALYZE")
                conn.commit()
                
                analyze_time = time.time() - start_time
                self.optimization_stats["analyze_operations"] += 1
                
                logger.info(f"✅ Analysis completed in {analyze_time:.2f}s")
                
                # Update statistics
                for stats in self.table_stats[db_name].values():
                    stats.last_analyzed = datetime.now()
            
        except Exception as e:
            logger.error(f"Error analyzing database {db_name}: {e}")
    
    async def _optimize_queries(self, db_name: str):
        """Optimize common query patterns"""
        try:
            # Identify and optimize common slow queries
            # This would involve query plan analysis and index suggestions
            pass
            
        except Exception as e:
            logger.error(f"Error optimizing queries for {db_name}: {e}")
    
    async def _suggest_missing_indexes(self, db_name: str):
        """Suggest missing indexes based on query patterns"""
        try:
            # Analyze common WHERE clauses and JOIN conditions
            # Suggest indexes for frequently queried columns
            pass
            
        except Exception as e:
            logger.error(f"Error suggesting missing indexes for {db_name}: {e}")
    
    async def _cleanup_unused_indexes(self, db_name: str):
        """Remove indexes that are never used"""
        try:
            unused_indexes = []
            
            for index_name, index_info in self.index_registry[db_name].items():
                if (index_info.usage_count == 0 and 
                    index_info.created_at < datetime.now() - timedelta(days=30)):
                    unused_indexes.append(index_name)
            
            for index_name in unused_indexes:
                self._drop_index(db_name, index_name)
            
        except Exception as e:
            logger.error(f"Error cleaning up unused indexes for {db_name}: {e}")
    
    def _drop_index(self, db_name: str, index_name: str):
        """Drop an unused index"""
        try:
            conn = self.connections[db_name]
            conn.execute(f"DROP INDEX IF EXISTS {index_name}")
            conn.commit()
            
            if index_name in self.index_registry[db_name]:
                del self.index_registry[db_name][index_name]
            
            logger.info(f"🗑️ Dropped unused index: {index_name}")
            
        except Exception as e:
            logger.error(f"Error dropping index {index_name}: {e}")
    
    def get_optimization_statistics(self) -> Dict[str, Any]:
        """Get comprehensive optimization statistics"""
        stats = {
            "optimization_stats": self.optimization_stats.copy(),
            "database_count": len(self.connections),
            "total_indexes": sum(len(indexes) for indexes in self.index_registry.values()),
            "database_details": {}
        }
        
        for db_name in self.connections.keys():
            db_stats = {
                "table_count": len(self.table_stats[db_name]),
                "index_count": len(self.index_registry[db_name]),
                "total_rows": sum(stats.row_count for stats in self.table_stats[db_name].values()),
                "total_size_mb": sum(stats.size_kb for stats in self.table_stats[db_name].values()) / 1024,
                "tables": {
                    table_name: {
                        "row_count": stats.row_count,
                        "size_kb": stats.size_kb,
                        "avg_row_size": stats.avg_row_size,
                        "fragmentation_percent": stats.fragmentation_percent
                    }
                    for table_name, stats in self.table_stats[db_name].items()
                }
            }
            stats["database_details"][db_name] = db_stats
        
        return stats
    
    def force_optimization(self, db_name: Optional[str] = None):
        """Force immediate optimization of specified database or all databases"""
        if db_name:
            if db_name in self.connections:
                asyncio.create_task(self._optimize_database(db_name))
            else:
                raise ValueError(f"Database not found: {db_name}")
        else:
            for db_name in self.connections.keys():
                asyncio.create_task(self._optimize_database(db_name))
    
    async def shutdown(self):
        """Shutdown database optimizer"""
        logger.info("🛑 Shutting down Database Optimizer")
        
        self.monitoring_active = False
        
        # Cancel background tasks
        if self.monitoring_task:
            self.monitoring_task.cancel()
        if self.optimization_task:
            self.optimization_task.cancel()
        
        # Final optimization before shutdown
        for db_name in self.connections.keys():
            try:
                conn = self.connections[db_name]
                conn.execute("PRAGMA optimize")
                conn.commit()
            except Exception as e:
                logger.error(f"Error during final optimization of {db_name}: {e}")
        
        # Close all connections
        for db_name, conn in self.connections.items():
            try:
                conn.close()
            except Exception as e:
                logger.error(f"Error closing connection to {db_name}: {e}")
        
        # Shutdown thread pool
        self.executor.shutdown(wait=True, timeout=30)
        
        logger.info("✅ Database Optimizer shutdown complete")

# Create singleton instance with default database paths
default_db_paths = {
    "main": "/data/queentrack.db",
    "auth": "/data/auth.db",
    "backup": "/data/backups/backup_db.sqlite",
    "compression": "/data/videos/compression_db.sqlite"
}

database_optimizer = DatabaseOptimizer(default_db_paths) 