"""Database management module for consciousness analysis"""

import sqlite3
import threading
from contextlib import contextmanager
from typing import Optional, List, Any, Dict
import logging

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Manages SQLite connections with proper locking and WAL mode"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._initialize_database()

    def _initialize_database(self):
        """Initialize database with WAL mode for better concurrency"""
        with self.get_connection() as conn:
            # Enable WAL mode for better concurrent access
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=10000")  # 10 second timeout
            conn.execute("PRAGMA synchronous=NORMAL")

            # Create all tables
            conn.executescript(self._get_schema())
            conn.commit()

    def _get_schema(self) -> str:
        """Get database schema SQL"""
        return """
            -- Navigator tables
            CREATE TABLE IF NOT EXISTS navigators (
                navigator_id TEXT PRIMARY KEY,
                email TEXT UNIQUE NOT NULL,
                phone TEXT,
                experiences TEXT NOT NULL,
                max_hours_per_week INTEGER DEFAULT 35,
                unavailable_topics TEXT DEFAULT '[]',
                support_group_id TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                CHECK (max_hours_per_week <= 40)
            );

            CREATE TABLE IF NOT EXISTS consciousness_capacity (
                navigator_id TEXT PRIMARY KEY REFERENCES navigators(navigator_id),
                complexity_scale TEXT NOT NULL CHECK (
                    complexity_scale IN ('component', 'subsystem', 'system', 'ecosystem')
                ),
                conscious_domains TEXT NOT NULL,
                fragmentation_resistance INTEGER NOT NULL CHECK (fragmentation_resistance BETWEEN 1 AND 10),
                consciousness_sharing_ability INTEGER NOT NULL CHECK (consciousness_sharing_ability BETWEEN 1 AND 10),
                pattern_recognition_depth INTEGER NOT NULL CHECK (pattern_recognition_depth BETWEEN 1 AND 10),
                care_drivers TEXT NOT NULL,
                formative_experiences TEXT NOT NULL,
                consciousness_stage TEXT DEFAULT 'emerging' CHECK (
                    consciousness_stage IN ('emerging', 'developing', 'integrated', 'mastery')
                )
            );

            CREATE TABLE IF NOT EXISTS time_violence_incidents (
                incident_id TEXT PRIMARY KEY DEFAULT (lower(hex(randomblob(16)))),
                navigator_id TEXT REFERENCES navigators(navigator_id),
                tweet_id TEXT,
                timestamp TEXT,
                domain TEXT,
                description TEXT,
                estimated_hours REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS consciousness_indicators (
                indicator_id TEXT PRIMARY KEY DEFAULT (lower(hex(randomblob(16)))),
                navigator_id TEXT REFERENCES navigators(navigator_id),
                tweet_id TEXT,
                timestamp TEXT,
                complexity_domain TEXT,
                consciousness_type TEXT,
                intensity_score REAL,
                situation_hash TEXT,
                content TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            -- Processing status tracking
            CREATE TABLE IF NOT EXISTS processing_status (
                username TEXT PRIMARY KEY,
                status TEXT CHECK (status IN ('pending', 'processing', 'completed', 'failed', 'skipped')),
                navigator_id TEXT,
                processed_at TIMESTAMP,
                tweet_count INTEGER,
                error_message TEXT,
                retry_count INTEGER DEFAULT 0,
                last_tweet_id TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            -- Community patterns
            CREATE TABLE IF NOT EXISTS community_patterns (
                pattern_id TEXT PRIMARY KEY DEFAULT (lower(hex(randomblob(16)))),
                pattern_hash TEXT UNIQUE,
                pattern_type TEXT,
                domain TEXT,
                occurrence_count INTEGER DEFAULT 1,
                navigators_demonstrating TEXT,
                first_seen TIMESTAMP,
                last_seen TIMESTAMP,
                pattern_description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            -- Collective metrics
            CREATE TABLE IF NOT EXISTS collective_metrics (
                metric_date DATE PRIMARY KEY,
                total_navigators INTEGER,
                total_time_violence_hours REAL,
                total_consciousness_indicators INTEGER,
                domains_covered TEXT,
                avg_fragmentation_resistance REAL,
                avg_consciousness_sharing REAL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            -- Create indices
            CREATE INDEX IF NOT EXISTS idx_processing_status ON processing_status(status);
            CREATE INDEX IF NOT EXISTS idx_processing_updated ON processing_status(updated_at);
            CREATE INDEX IF NOT EXISTS idx_patterns_domain ON community_patterns(domain);
            CREATE INDEX IF NOT EXISTS idx_patterns_count ON community_patterns(occurrence_count DESC);
        """

    @contextmanager
    def get_connection(self):
        """Get a database connection with proper management"""
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def execute_query(self, query: str, params: Optional[tuple] = None) -> List[sqlite3.Row]:
        """Execute a query and return results"""
        with self.get_connection() as conn:
            cur = conn.cursor()
            if params:
                cur.execute(query, params)
            else:
                cur.execute(query)
            return cur.fetchall()

    def execute_write(self, query: str, params: Optional[tuple] = None) -> Optional[int]:
        """Execute a write operation with proper transaction handling"""
        with self.get_connection() as conn:
            cur = conn.cursor()
            try:
                if params:
                    cur.execute(query, params)
                else:
                    cur.execute(query)
                conn.commit()
                return cur.lastrowid
            except sqlite3.Error as e:
                conn.rollback()
                raise e

    def execute_many(self, query: str, params_list: List[tuple]):
        """Execute many write operations in a single transaction"""
        with self.get_connection() as conn:
            cur = conn.cursor()
            try:
                cur.executemany(query, params_list)
                conn.commit()
            except sqlite3.Error as e:
                conn.rollback()
                raise e

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics from the database"""
        with self.get_connection() as conn:
            cur = conn.cursor()

            stats = {}

            # Total navigators
            cur.execute("SELECT COUNT(*) FROM navigators")
            stats['total_navigators'] = cur.fetchone()[0]

            # Consciousness distribution
            cur.execute("""
                SELECT consciousness_stage, COUNT(*)
                FROM consciousness_capacity
                GROUP BY consciousness_stage
            """)
            stats['consciousness_stages'] = dict(cur.fetchall())

            # Total Time Violence
            cur.execute("SELECT SUM(estimated_hours) FROM time_violence_incidents")
            result = cur.fetchone()[0]
            stats['total_time_violence_hours'] = result if result else 0

            # Top domains
            cur.execute("""
                SELECT domain, COUNT(*), SUM(estimated_hours)
                FROM time_violence_incidents
                GROUP BY domain
                ORDER BY SUM(estimated_hours) DESC
                LIMIT 5
            """)
            stats['top_time_violence_domains'] = [
                {'domain': row[0], 'incidents': row[1], 'hours': row[2]}
                for row in cur.fetchall()
            ]

            return stats