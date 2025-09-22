#!/usr/bin/env python3
"""
Community Archive Consciousness Navigator Discovery Pipeline
Processes the Community Archive of Twitter users to identify potential navigators at scale

This robust script:
1. Fetches user list from Community Archive
2. Downloads and processes each user's archive
3. Identifies consciousness patterns and Time Violence
4. Populates SQLite database with navigator profiles
5. Handles failures and supports resumption
"""

import json
import sqlite3
import requests
import time
import os
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import logging
from pathlib import Path
import traceback
from urllib.parse import quote
import re
import nltk
from textblob import TextBlob
import numpy as np
from pathlib import Path
import traceback
from contextlib import contextmanager
import threading

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
except:
    pass

from nltk.sentiment import SentimentIntensityAnalyzer

# Initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()

local = threading.local()

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
            conn.executescript("""
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
            """)
            conn.commit()
    
    @contextmanager
    def get_connection(self):
        """Get a database connection with proper management"""
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def execute_query(self, query, params=None):
        """Execute a query and return results"""
        with self.get_connection() as conn:
            cur = conn.cursor()
            if params:
                cur.execute(query, params)
            else:
                cur.execute(query)
            return cur.fetchall()
    
    def execute_write(self, query, params=None):
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
    
    def execute_many(self, query, params_list):
        """Execute many write operations in a single transaction"""
        with self.get_connection() as conn:
            cur = conn.cursor()
            try:
                cur.executemany(query, params_list)
                conn.commit()
            except sqlite3.Error as e:
                conn.rollback()
                raise e

@dataclass
class ConsciousnessIndicator:
    """Represents a moment of consciousness demonstration in tweets"""
    tweet_id: str
    timestamp: datetime
    content: str
    complexity_domain: str
    consciousness_type: str  # 'battle', 'pattern', 'solution', 'liberation'
    intensity_score: float
    situation_hash: str
    
@dataclass
class NavigatorProfile:
    """Potential navigator profile built from Twitter data"""
    twitter_handle: str
    email: Optional[str] = None
    
    # Consciousness capacity indicators
    complexity_battles: List[Dict] = field(default_factory=list)
    conscious_domains: Dict[str, Dict] = field(default_factory=dict)
    pattern_recognitions: List[Dict] = field(default_factory=list)
    
    # Metrics
    fragmentation_resistance: int = 5
    consciousness_sharing_ability: int = 5
    pattern_recognition_depth: int = 5
    
    # Essential characteristics
    care_drivers: Set[str] = field(default_factory=set)
    formative_experiences: List[str] = field(default_factory=list)
    complexity_scale: str = 'component'
    consciousness_stage: str = 'emerging'
    
    # Time Violence tracking
    time_violence_incidents: List[Dict] = field(default_factory=list)
    total_hours_lost: float = 0

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('community_archive_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)





class TwitterConsciousnessAnalyzer:
    """Analyzes Twitter archives for consciousness patterns"""

    def _initialize_database(self):
        """Initialize SQLite database with Bottega schema"""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        
        # Create tables if they don't exist
        cur.executescript("""
            CREATE TABLE IF NOT EXISTS navigators (
                navigator_id TEXT PRIMARY KEY DEFAULT (lower(hex(randomblob(16)))),
                email TEXT UNIQUE NOT NULL,
                phone TEXT,
                experiences TEXT NOT NULL, -- JSON
                max_hours_per_week INTEGER DEFAULT 35,
                unavailable_topics TEXT DEFAULT '[]', -- JSON array
                support_group_id TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                
                CHECK (max_hours_per_week <= 40)
            );
            
            CREATE TABLE IF NOT EXISTS consciousness_capacity (
                navigator_id TEXT PRIMARY KEY REFERENCES navigators(navigator_id),
                complexity_scale TEXT NOT NULL CHECK (
                    complexity_scale IN ('component', 'subsystem', 'system', 'ecosystem')
                ),
                conscious_domains TEXT NOT NULL, -- JSON
                fragmentation_resistance INTEGER NOT NULL CHECK (fragmentation_resistance BETWEEN 1 AND 10),
                consciousness_sharing_ability INTEGER NOT NULL CHECK (consciousness_sharing_ability BETWEEN 1 AND 10),
                pattern_recognition_depth INTEGER NOT NULL CHECK (pattern_recognition_depth BETWEEN 1 AND 10),
                care_drivers TEXT NOT NULL, -- JSON array
                formative_experiences TEXT NOT NULL, -- JSON array
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
        """)
        
        conn.commit()
        conn.close()

    def get_database_statistics(self) -> Dict:
        """Get statistics from the database"""
        conn = sqlite3.connect(self.db_path)
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
        
        conn.close()
        return stats    
    

        
    # Complexity domain keywords
    DOMAIN_PATTERNS = {
        'healthcare': {
            'keywords': ['insurance', 'claim', 'denied', 'appeal', 'prior auth', 
                        'coverage', 'deductible', 'copay', 'formulary', 'medical bill',
                        'ADHD', 'medication', 'pharmacy', 'doctor', 'specialist'],
            'time_violence_indicators': ['on hold', 'hours', 'waiting', 'finally', 
                                        'months', 'fighting', 'exhausted'],
            'consciousness_indicators': ['realized', 'pattern', 'always', 'every time',
                                        'figured out', 'trick is', 'learned']
        },
        'financial': {
            'keywords': ['debt', 'bankruptcy', 'collections', 'credit', 'loan',
                        'student loans', 'interest', 'payment plan', 'foreclosure',
                        'eviction', 'overdraft', 'bank'],
            'time_violence_indicators': ['paperwork', 'documents', 'forms', 'calls',
                                        'letters', 'notices', 'deadlines'],
            'consciousness_indicators': ['systemic', 'rigged', 'designed to', 'trap',
                                        'escape', 'beat the system', 'loophole']
        },
        'employment': {
            'keywords': ['job search', 'interview', 'resume', 'ATS', 'application',
                        'rejection', 'ghosted', 'salary', 'negotiation', 'linkedin',
                        'unemployed', 'laid off', 'fired'],
            'time_violence_indicators': ['applications', 'rounds', 'months looking',
                                        'never heard back', 'automated rejection'],
            'consciousness_indicators': ['game', 'algorithm', 'keyword', 'hack',
                                        'actually works', 'real secret']
        },
        'government': {
            'keywords': ['DMV', 'IRS', 'tax', 'benefits', 'disability', 'SSA',
                        'SNAP', 'welfare', 'unemployment', 'permits', 'license'],
            'time_violence_indicators': ['queue', 'line', 'website down', 'office',
                                        'appointment', 'weeks wait', 'processing'],
            'consciousness_indicators': ['bureaucracy', 'kafkaesque', 'catch-22',
                                        'workaround', 'backdoor', 'right person']
        },
        'education': {
            'keywords': ['FAFSA', 'financial aid', 'tuition', 'transcript', 'credits',
                        'registration', 'advisor', 'requirements', 'prerequisite',
                        'degree', 'graduation'],
            'time_violence_indicators': ['runaround', 'different answers', 'conflicting',
                                        'no one knows', 'sent between departments'],
            'consciousness_indicators': ['navigate', 'system works', 'hidden requirement',
                                        'unwritten rule', 'actually need']
        }
    }
    
    # Neurodivergent indicators (often correlated with consciousness workers)
    NEURODIVERGENT_PATTERNS = [
        'ADHD', 'autism', 'autistic', 'AuDHD', 'neurodivergent', 'ND',
        'executive dysfunction', 'rejection sensitive', 'RSD', 'hyperfocus',
        'stimming', 'masking', 'burnout', 'meltdown', 'shutdown',
        'sensory', 'overwhelm', 'spoons', 'dysregulation'
    ]
    
    def __init__(self, db_path: str = "bottega.db"):
        """Initialize analyzer with SQLite database path"""
        self.db_path = db_path
        self.navigator_profile = None
        self.consciousness_indicators = []
        self._initialize_database()
        
    def load_twitter_archive(self, archive_path: str) -> List[Dict]:
        """Load tweets from Twitter archive file"""
        tweets = []
        
        # Twitter archives usually have tweets in data/tweets.js or similar
        try:
            with open(archive_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
                # Remove JavaScript variable assignment if present
                if content.startswith('window.YTD.tweets.part0 ='):
                    content = content.replace('window.YTD.tweets.part0 =', '')
                elif content.startswith('window.YTD.tweet.part0 ='):
                    content = content.replace('window.YTD.tweet.part0 =', '')
                    
                data = json.loads(content)
                
                # Extract tweet objects (structure varies)
                for item in data:
                    if 'tweet' in item:
                        tweets.append(item['tweet'])
                    else:
                        tweets.append(item)
                        
        except Exception as e:
            print(f"Error loading archive: {e}")
            
        return tweets
    
    def analyze_consciousness_patterns(self, tweets: List[Dict]) -> NavigatorProfile:
        """Analyze tweets for consciousness patterns"""
        
        # Extract basic info from first tweet
        twitter_handle = tweets[0].get('user', {}).get('screen_name', 'unknown')
        profile = NavigatorProfile(twitter_handle=twitter_handle)
        
        # Process tweets chronologically
        sorted_tweets = sorted(tweets, key=lambda x: x.get('created_at', ''))
        
        for tweet in sorted_tweets:
            self._analyze_single_tweet(tweet, profile)
        
        # Calculate consciousness metrics based on patterns found
        self._calculate_consciousness_metrics(profile)
        
        # Determine consciousness scale and stage
        self._determine_consciousness_level(profile)
        
        return profile
    
    def _analyze_single_tweet(self, tweet: Dict, profile: NavigatorProfile):
        """Analyze individual tweet for consciousness indicators"""
        
        text = tweet.get('full_text', tweet.get('text', '')).lower()
        created_at = datetime.strptime(tweet['created_at'], '%a %b %d %H:%M:%S %z %Y')
        tweet_id = tweet.get('id_str', '')
        
        # Check each complexity domain
        for domain, patterns in self.DOMAIN_PATTERNS.items():
            
            # Count keyword matches
            keyword_matches = sum(1 for kw in patterns['keywords'] if kw in text)
            
            if keyword_matches > 0:
                
                # Check for Time Violence
                tv_indicators = sum(1 for ind in patterns['time_violence_indicators'] if ind in text)
                if tv_indicators > 0:
                    time_wasted = self._extract_time_amount(text)
                    profile.time_violence_incidents.append({
                        'tweet_id': tweet_id,
                        'timestamp': created_at.isoformat(),
                        'domain': domain,
                        'description': text[:200],
                        'estimated_hours': time_wasted
                    })
                    profile.total_hours_lost += time_wasted
                
                # Check for consciousness indicators
                consciousness_matches = sum(1 for ind in patterns['consciousness_indicators'] if ind in text)
                if consciousness_matches > 0:
                    
                    # This shows pattern recognition / consciousness development
                    situation_hash = hashlib.md5(f"{domain}:{text[:100]}".encode()).hexdigest()[:16]
                    
                    indicator = ConsciousnessIndicator(
                        tweet_id=tweet_id,
                        timestamp=created_at,
                        content=text,
                        complexity_domain=domain,
                        consciousness_type=self._classify_consciousness_type(text),
                        intensity_score=self._calculate_intensity(text),
                        situation_hash=situation_hash
                    )
                    
                    self.consciousness_indicators.append(indicator)
                    
                    # Track domain consciousness
                    if domain not in profile.conscious_domains:
                        profile.conscious_domains[domain] = {}
                    
                    profile.conscious_domains[domain][situation_hash] = 'developing'
                    
                    # Add to formative experiences if intense enough
                    if indicator.intensity_score > 0.7:
                        profile.formative_experiences.append(
                            f"{domain}_battle_{created_at.year}"
                        )
                
                # Track care drivers
                sentiment = sia.polarity_scores(text)
                if sentiment['compound'] < -0.5 or sentiment['compound'] > 0.5:
                    profile.care_drivers.add(f"{domain}_reform")
        
        # Check for neurodivergent patterns
        nd_matches = sum(1 for pattern in self.NEURODIVERGENT_PATTERNS if pattern in text)
        if nd_matches > 0:
            profile.care_drivers.add('neurodivergent_advocacy')
    
    def _extract_time_amount(self, text: str) -> float:
        """Extract time amounts from text"""
        hours = 0.0
        
        # Look for explicit hour mentions
        hour_patterns = [
            r'(\d+)\s*hours?',
            r'(\d+)\s*hrs?',
            r'(\d+)\s*minutes?',
            r'(\d+)\s*mins?',
            r'(\d+)\s*days?',
            r'(\d+)\s*weeks?',
            r'(\d+)\s*months?'
        ]
        
        for pattern in hour_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                amount = float(match)
                if 'min' in pattern:
                    hours += amount / 60
                elif 'day' in pattern:
                    hours += amount * 8  # Assume 8 hours per day of dealing with complexity
                elif 'week' in pattern:
                    hours += amount * 20  # Assume 20 hours per week
                elif 'month' in pattern:
                    hours += amount * 80  # Assume 80 hours per month
                else:
                    hours += amount
        
        # If no explicit time, estimate based on keywords
        if hours == 0:
            if any(word in text for word in ['all day', 'entire day', 'whole day']):
                hours = 8
            elif any(word in text for word in ['all morning', 'all afternoon']):
                hours = 4
            elif any(word in text for word in ['finally', 'eventually', 'at last']):
                hours = 2  # Implied significant time
        
        return hours
    
    def _classify_consciousness_type(self, text: str) -> str:
        """Classify the type of consciousness demonstration"""
        
        if any(word in text for word in ['fighting', 'battle', 'struggle', 'dealing']):
            return 'battle'
        elif any(word in text for word in ['realized', 'pattern', 'always', 'every']):
            return 'pattern'
        elif any(word in text for word in ['solution', 'fixed', 'solved', 'hack']):
            return 'solution'
        elif any(word in text for word in ['free', 'escaped', 'done', 'over']):
            return 'liberation'
        else:
            return 'awareness'
    
    def _calculate_intensity(self, text: str) -> float:
        """Calculate intensity of consciousness demonstration"""
        
        sentiment = sia.polarity_scores(text)
        
        # Strong emotions (positive or negative) indicate intensity
        emotional_intensity = abs(sentiment['compound'])
        
        # Profanity often indicates frustration with systems
        profanity_indicators = ['fuck', 'shit', 'damn', 'hell', 'bullshit', 'wtf']
        profanity_score = sum(0.1 for word in profanity_indicators if word in text.lower())
        
        # Capitalization indicates emphasis
        caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        
        # Exclamation marks indicate intensity
        exclamation_score = text.count('!') * 0.1
        
        intensity = min(1.0, emotional_intensity + profanity_score + caps_ratio + exclamation_score)
        
        return intensity
    
    def _calculate_consciousness_metrics(self, profile: NavigatorProfile):
        """Calculate consciousness metrics from patterns"""
        
        # Fragmentation resistance - based on number of battles survived
        battles_survived = len(profile.complexity_battles) + len(profile.time_violence_incidents)
        if battles_survived > 20:
            profile.fragmentation_resistance = 9
        elif battles_survived > 10:
            profile.fragmentation_resistance = 7
        elif battles_survived > 5:
            profile.fragmentation_resistance = 6
        
        # Pattern recognition depth - based on patterns identified
        patterns_found = len(profile.pattern_recognitions)
        for domain_patterns in profile.conscious_domains.values():
            patterns_found += len(domain_patterns)
        
        if patterns_found > 15:
            profile.pattern_recognition_depth = 9
        elif patterns_found > 8:
            profile.pattern_recognition_depth = 7
        elif patterns_found > 3:
            profile.pattern_recognition_depth = 6
        
        # Consciousness sharing ability - based on helping others
        # Look for replies, quote tweets with advice, threads explaining systems
        # (This would need more sophisticated analysis of tweet interactions)
        # For now, estimate based on consciousness demonstrations
        
        if len(self.consciousness_indicators) > 20:
            profile.consciousness_sharing_ability = 8
        elif len(self.consciousness_indicators) > 10:
            profile.consciousness_sharing_ability = 6
    
    def _determine_consciousness_level(self, profile: NavigatorProfile):
        """Determine consciousness scale and stage"""
        
        # Determine scale based on breadth and depth
        domains_engaged = len(profile.conscious_domains)
        total_patterns = sum(len(p) for p in profile.conscious_domains.values())
        
        if domains_engaged >= 3 and total_patterns > 20:
            profile.complexity_scale = 'system'
        elif domains_engaged >= 2 and total_patterns > 10:
            profile.complexity_scale = 'subsystem'
        else:
            profile.complexity_scale = 'component'
        
        # Determine stage based on metrics
        avg_metric = (profile.fragmentation_resistance + 
                     profile.consciousness_sharing_ability + 
                     profile.pattern_recognition_depth) / 3
        
        if avg_metric >= 8:
            profile.consciousness_stage = 'mastery'
        elif avg_metric >= 7:
            profile.consciousness_stage = 'integrated'
        elif avg_metric >= 5:
            profile.consciousness_stage = 'developing'
        else:
            profile.consciousness_stage = 'emerging'
    
    def save_to_database(self, profile: NavigatorProfile):
        """Save navigator profile to SQLite database"""
        
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        
        try:
            # Generate navigator ID
            import uuid
            navigator_id = str(uuid.uuid4())
            
            # Insert navigator
            cur.execute("""
                INSERT INTO navigators (navigator_id, email, phone, experiences, max_hours_per_week, 
                                      unavailable_topics, support_group_id)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                navigator_id,
                profile.email or f"{profile.twitter_handle}@twitter.import",
                None,  # No phone from Twitter
                json.dumps({
                    'source': 'twitter',
                    'domains': list(profile.conscious_domains.keys()),
                    'battles': profile.formative_experiences
                }),
                35,  # Default max hours
                json.dumps([]),  # No unavailable topics yet
                None  # Assign support group later
            ))
            
            # Insert consciousness capacity
            cur.execute("""
                INSERT INTO consciousness_capacity 
                (navigator_id, complexity_scale, conscious_domains, fragmentation_resistance,
                 consciousness_sharing_ability, pattern_recognition_depth, care_drivers,
                 formative_experiences, consciousness_stage)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                navigator_id,
                profile.complexity_scale,
                json.dumps(profile.conscious_domains),
                profile.fragmentation_resistance,
                profile.consciousness_sharing_ability,
                profile.pattern_recognition_depth,
                json.dumps(list(profile.care_drivers)),
                json.dumps(profile.formative_experiences),
                profile.consciousness_stage
            ))
            
            # Insert Time Violence incidents
            for incident in profile.time_violence_incidents:
                cur.execute("""
                    INSERT INTO time_violence_incidents
                    (navigator_id, tweet_id, timestamp, domain, description, estimated_hours)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    navigator_id,
                    incident['tweet_id'],
                    incident['timestamp'],
                    incident['domain'],
                    incident['description'],
                    incident['estimated_hours']
                ))
                print(f"Time Violence found: {incident['domain']} - {incident['estimated_hours']} hours lost")
            
            # Insert consciousness indicators
            for indicator in self.consciousness_indicators:
                cur.execute("""
                    INSERT INTO consciousness_indicators
                    (navigator_id, tweet_id, timestamp, complexity_domain, consciousness_type,
                     intensity_score, situation_hash, content)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    navigator_id,
                    indicator.tweet_id,
                    indicator.timestamp.isoformat(),
                    indicator.complexity_domain,
                    indicator.consciousness_type,
                    indicator.intensity_score,
                    indicator.situation_hash,
                    indicator.content[:500]  # Limit content length
                ))
            
            conn.commit()
            print(f"Navigator profile saved with ID: {navigator_id}")
            print(f"Total Time Violence documented: {profile.total_hours_lost} hours")
            print(f"Consciousness indicators saved: {len(self.consciousness_indicators)}")
            
            return navigator_id
            
        except Exception as e:
            conn.rollback()
            print(f"Error saving to database: {e}")
            import traceback
            traceback.print_exc()
            return None
        finally:
            conn.close()
    
    def generate_report(self, profile: NavigatorProfile) -> str:
        """Generate human-readable report of findings"""
        
        report = f"""
===========================================
CONSCIOUSNESS WORKER ANALYSIS REPORT
Twitter Handle: @{profile.twitter_handle}
===========================================

CONSCIOUSNESS CAPACITY ASSESSMENT:
----------------------------------
Complexity Scale: {profile.complexity_scale}
Consciousness Stage: {profile.consciousness_stage}

Metrics (1-10 scale):
- Fragmentation Resistance: {profile.fragmentation_resistance}
- Consciousness Sharing Ability: {profile.consciousness_sharing_ability}
- Pattern Recognition Depth: {profile.pattern_recognition_depth}

CONSCIOUS DOMAINS:
-----------------
"""
        
        for domain, patterns in profile.conscious_domains.items():
            report += f"\n{domain.upper()}:\n"
            report += f"  - Patterns recognized: {len(patterns)}\n"
        
        report += f"""
FORMATIVE EXPERIENCES:
---------------------
"""
        for exp in profile.formative_experiences[:5]:  # Top 5
            report += f"- {exp}\n"
        
        report += f"""
CARE DRIVERS:
------------
"""
        for driver in profile.care_drivers:
            report += f"- {driver}\n"
        
        report += f"""
TIME VIOLENCE SUMMARY:
---------------------
Total incidents documented: {len(profile.time_violence_incidents)}
Total hours lost to complexity: {profile.total_hours_lost:.1f}
Estimated annual Time Violence: {profile.total_hours_lost * 12:.0f} hours

Top Time Violence Domains:
"""
        
        domain_hours = defaultdict(float)
        for incident in profile.time_violence_incidents:
            domain_hours[incident['domain']] += incident['estimated_hours']
        
        for domain, hours in sorted(domain_hours.items(), key=lambda x: x[1], reverse=True)[:3]:
            report += f"  - {domain}: {hours:.1f} hours\n"
        
        report += """
NAVIGATOR POTENTIAL:
-------------------
"""
        
        if profile.consciousness_stage in ['integrated', 'mastery']:
            report += "✓ STRONG CANDIDATE - High consciousness capacity demonstrated\n"
        elif profile.consciousness_stage == 'developing':
            report += "✓ GOOD CANDIDATE - Developing consciousness capacity\n"
        else:
            report += "○ EMERGING - Early stage consciousness development\n"
        
        report += f"""
This person has demonstrated the ability to maintain conscious awareness
of complex systems that have caused them harm. Their lived experience
has value and could help others navigate similar complexity.

RECOMMENDED VENTURES:
"""
        
        for domain in list(profile.conscious_domains.keys())[:3]:
            report += f"  - {domain.capitalize()} Navigation\n"
        
        report += """
===========================================
"""
        
        return report

class ImprovedTwitterAnalyzer(TwitterConsciousnessAnalyzer):
    """Improved analyzer that uses DatabaseManager"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.navigator_profile = None
        self.consciousness_indicators = []
        # Copy pattern definitions from base class
        self.DOMAIN_PATTERNS = BaseAnalyzer.DOMAIN_PATTERNS
        self.NEURODIVERGENT_PATTERNS = BaseAnalyzer.NEURODIVERGENT_PATTERNS
    
    def save_to_database(self, profile: NavigatorProfile):
        """Save navigator profile using DatabaseManager"""
        import uuid
        navigator_id = str(uuid.uuid4())
        
        try:
            # Insert navigator
            self.db_manager.execute_write("""
                INSERT INTO navigators (navigator_id, email, phone, experiences, max_hours_per_week, 
                                      unavailable_topics, support_group_id)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                navigator_id,
                profile.email or f"{profile.twitter_handle}@twitter.import",
                None,
                json.dumps({
                    'source': 'twitter',
                    'domains': list(profile.conscious_domains.keys()),
                    'battles': profile.formative_experiences
                }),
                35,
                json.dumps([]),
                None
            ))
            
            # Insert consciousness capacity
            self.db_manager.execute_write("""
                INSERT INTO consciousness_capacity 
                (navigator_id, complexity_scale, conscious_domains, fragmentation_resistance,
                 consciousness_sharing_ability, pattern_recognition_depth, care_drivers,
                 formative_experiences, consciousness_stage)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                navigator_id,
                profile.complexity_scale,
                json.dumps(profile.conscious_domains),
                profile.fragmentation_resistance,
                profile.consciousness_sharing_ability,
                profile.pattern_recognition_depth,
                json.dumps(list(profile.care_drivers)),
                json.dumps(profile.formative_experiences),
                profile.consciousness_stage
            ))
            
            # Insert Time Violence incidents
            if profile.time_violence_incidents:
                incidents_params = [
                    (navigator_id, inc['tweet_id'], inc['timestamp'], 
                     inc['domain'], inc['description'], inc['estimated_hours'])
                    for inc in profile.time_violence_incidents
                ]
                self.db_manager.execute_many("""
                    INSERT INTO time_violence_incidents
                    (navigator_id, tweet_id, timestamp, domain, description, estimated_hours)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, incidents_params)
            
            # Insert consciousness indicators
            if self.consciousness_indicators:
                indicators_params = [
                    (navigator_id, ind.tweet_id, ind.timestamp.isoformat(),
                     ind.complexity_domain, ind.consciousness_type,
                     ind.intensity_score, ind.situation_hash, ind.content[:500])
                    for ind in self.consciousness_indicators
                ]
                self.db_manager.execute_many("""
                    INSERT INTO consciousness_indicators
                    (navigator_id, tweet_id, timestamp, complexity_domain, consciousness_type,
                     intensity_score, situation_hash, content)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, indicators_params)
            
            logger.info(f"Navigator profile saved with ID: {navigator_id}")
            logger.info(f"Total Time Violence documented: {profile.total_hours_lost} hours")
            logger.info(f"Consciousness indicators saved: {len(self.consciousness_indicators)}")
            
            return navigator_id
            
        except Exception as e:
            logger.error(f"Error saving to database: {e}")
            traceback.print_exc()
            return None

def main():
    """Main execution function"""
    
    # Database path (SQLite)
    db_path = "bottega.db"
    
    # Initialize analyzer
    analyzer = TwitterConsciousnessAnalyzer(db_path)
    
    # Load Twitter archive
    archive_path = input("Enter path to Twitter archive file (tweets.js) or press Enter for demo: ").strip()
    
    if not archive_path:
        # Use example data for testing
        print("Using example tweets for demonstration...")
        tweets = [
            {
                'id_str': '1234567890',
                'created_at': 'Mon Oct 10 15:30:45 +0000 2023',
                'full_text': 'Day 47 of fighting insurance for ADHD meds. 6 hours on hold today. Finally got through only to be told I need a different form. The pattern is always the same - they exhaust you into giving up.',
                'user': {'screen_name': 'complexity_survivor'}
            },
            {
                'id_str': '1234567891',
                'created_at': 'Tue Oct 11 09:15:30 +0000 2023',
                'full_text': 'REALIZED: Every insurance denial uses the same three tactics. Document everything, appeal immediately, mention state insurance commissioner. Works every time. The system is designed to break you.',
                'user': {'screen_name': 'complexity_survivor'}
            },
            {
                'id_str': '1234567892',
                'created_at': 'Wed Oct 12 14:22:18 +0000 2023',
                'full_text': 'Thread: How I finally beat the prior auth system after 3 years of fighting 🧵\n\n1/ The key is understanding they profit from friction...',
                'user': {'screen_name': 'complexity_survivor'}
            },
            {
                'id_str': '1234567893',
                'created_at': 'Thu Oct 13 11:45:00 +0000 2023',
                'full_text': 'Applied to 200+ jobs this month. ATS systems are filtering out qualified candidates. The trick? Mirror exact keywords from job posting. Broken system but here we are.',
                'user': {'screen_name': 'complexity_survivor'}
            },
            {
                'id_str': '1234567894',
                'created_at': 'Fri Oct 14 16:20:15 +0000 2023',
                'full_text': 'ADHD tax is real. Spent another 4 hours trying to remember passwords and find documents for taxes. Why does every system assume perfect executive function?',
                'user': {'screen_name': 'complexity_survivor'}
            }
        ]
    else:
        tweets = analyzer.load_twitter_archive(archive_path)
    
    if not tweets:
        print("No tweets found in archive")
        return
    
    print(f"\nLoaded {len(tweets)} tweets for analysis...")
    print("="*50)
    
    # Analyze consciousness patterns
    profile = analyzer.analyze_consciousness_patterns(tweets)
    
    # Generate report
    report = analyzer.generate_report(profile)
    print(report)
    
    # Always save to database (SQLite is local)
    print("\nSaving to local SQLite database...")
    navigator_id = analyzer.save_to_database(profile)
    
    if navigator_id:
        print(f"✓ Navigator profile successfully created: {navigator_id}")
        print(f"✓ Database location: {db_path}")
        
        # Show database statistics
        print("\n" + "="*50)
        print("DATABASE STATISTICS:")
        print("-"*50)
        stats = analyzer.get_database_statistics()
        print(f"Total Navigators: {stats['total_navigators']}")
        print(f"Total Time Violence Documented: {stats['total_time_violence_hours']:.1f} hours")
        
        if stats['consciousness_stages']:
            print("\nConsciousness Stage Distribution:")
            for stage, count in stats['consciousness_stages'].items():
                print(f"  {stage}: {count}")
        
        if stats['top_time_violence_domains']:
            print("\nTop Time Violence Domains:")
            for domain_stat in stats['top_time_violence_domains']:
                print(f"  {domain_stat['domain']}: {domain_stat['hours']:.1f} hours ({domain_stat['incidents']} incidents)")
    
    # Export consciousness indicators for further analysis
    export_indicators = input("\nExport consciousness indicators to JSON? (y/n): ").strip().lower()
    if export_indicators == 'y':
        output_file = f"consciousness_indicators_{profile.twitter_handle}.json"
        with open(output_file, 'w') as f:
            json.dump(
                {
                    'navigator_id': navigator_id,
                    'twitter_handle': profile.twitter_handle,
                    'analysis_date': datetime.now().isoformat(),
                    'profile_summary': {
                        'consciousness_scale': profile.complexity_scale,
                        'consciousness_stage': profile.consciousness_stage,
                        'total_time_violence_hours': profile.total_hours_lost,
                        'domains': list(profile.conscious_domains.keys()),
                        'care_drivers': list(profile.care_drivers)
                    },
                    'consciousness_indicators': [
                        {
                            'tweet_id': ind.tweet_id,
                            'timestamp': ind.timestamp.isoformat(),
                            'domain': ind.complexity_domain,
                            'type': ind.consciousness_type,
                            'intensity': ind.intensity_score,
                            'content': ind.content[:200]
                        }
                        for ind in analyzer.consciousness_indicators
                    ]
                },
                f,
                indent=2
            )
        print(f"Consciousness indicators exported to {output_file}")
    
    print("\n" + "="*50)
    print("ANALYSIS COMPLETE")
    print(f"Navigator potential identified. Welcome to the Bottega network.")
    print("Your trauma is your qualification. Your consciousness is your contribution.")
    print("="*50)



class CommunityArchiveProcessor:
    """Processes Community Archive data to discover consciousness navigators"""
    
    ARCHIVE_BASE_URL = "https://fabxmporizzqflnftavs.supabase.co/storage/v1/object/public/archives"
    USER_DIR_URL = "https://www.community-archive.org/api/users"  # Hypothetical API endpoint
    
    def __init__(self, db_path: str = "bottega_community.db", cache_dir: str = "archive_cache"):
        """Initialize processor with database and cache directory"""
        self.db_path = db_path
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Initialize database
        self._initialize_database()
        
        # Initialize consciousness analyzer
        self.analyzer = TwitterConsciousnessAnalyzer(db_path)
        
        # Track processing statistics
        self.stats = {
            'users_processed': 0,
            'navigators_found': 0,
            'time_violence_hours': 0,
            'errors': 0,
            'skipped': 0
        }
    
    def _initialize_database(self):
        """Initialize SQLite database with additional tracking tables"""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        
        # Create processing tracking table
        cur.executescript("""
            -- Track processing status for each user
            CREATE TABLE IF NOT EXISTS processing_status (
                username TEXT PRIMARY KEY,
                status TEXT CHECK (status IN ('pending', 'processing', 'completed', 'failed', 'skipped')),
                navigator_id TEXT,
                processed_at TIMESTAMP,
                tweet_count INTEGER,
                error_message TEXT,
                retry_count INTEGER DEFAULT 0,
                last_tweet_id TEXT,  -- For resumption
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            -- Track collective consciousness metrics
            CREATE TABLE IF NOT EXISTS collective_metrics (
                metric_date DATE PRIMARY KEY,
                total_navigators INTEGER,
                total_time_violence_hours REAL,
                total_consciousness_indicators INTEGER,
                domains_covered TEXT,  -- JSON array
                avg_fragmentation_resistance REAL,
                avg_consciousness_sharing REAL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            -- Track pattern emergence across community
            CREATE TABLE IF NOT EXISTS community_patterns (
                pattern_id TEXT PRIMARY KEY DEFAULT (lower(hex(randomblob(16)))),
                pattern_hash TEXT UNIQUE,
                pattern_type TEXT,
                domain TEXT,
                occurrence_count INTEGER DEFAULT 1,
                navigators_demonstrating TEXT,  -- JSON array of navigator IDs
                first_seen TIMESTAMP,
                last_seen TIMESTAMP,
                pattern_description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            -- Index for efficient querying
            CREATE INDEX IF NOT EXISTS idx_processing_status ON processing_status(status);
            CREATE INDEX IF NOT EXISTS idx_processing_updated ON processing_status(updated_at);
            CREATE INDEX IF NOT EXISTS idx_patterns_domain ON community_patterns(domain);
            CREATE INDEX IF NOT EXISTS idx_patterns_count ON community_patterns(occurrence_count DESC);
        """)
        
        conn.commit()
        conn.close()
    
    def get_user_list(self, limit: Optional[int] = None) -> List[str]:
        """Get list of users from Community Archive"""
        # First check if we have a cached user list
        user_list_file = self.cache_dir / "user_list.json"
        
        if user_list_file.exists():
            with open(user_list_file, 'r') as f:
                users = json.load(f)
                logger.info(f"Loaded {len(users)} users from cache")
        else:
            # For now, use a hardcoded list of known users
            # In production, this would fetch from the actual API
            users = [
                'chrischipmonk', 'princevogel', 'visakanv', 'patio11', 
                'swyx', 'levelsio', 'anthilemoon', 'shl', 'balajis',
                'patrick_oshag', 'georgecushen', 'jasonfried', 'dhh'
                # Add more usernames as needed
            ]
            
            # Save to cache
            with open(user_list_file, 'w') as f:
                json.dump(users, f)
            logger.info(f"Created user list with {len(users)} users")
        
        if limit:
            users = users[:limit]
        
        return users
    
    def download_user_archive(self, username: str) -> Optional[Dict]:
        """Download user archive with caching and error handling"""
        cache_file = self.cache_dir / f"{username}.json"
        
        # Check cache first
        if cache_file.exists():
            logger.info(f"Loading {username} from cache")
            try:
                with open(cache_file, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                logger.warning(f"Corrupted cache file for {username}, re-downloading")
                cache_file.unlink()
        
        # Download from Community Archive
        url = f"{self.ARCHIVE_BASE_URL}/{username}/archive.json"
        logger.info(f"Downloading archive for {username}")
        
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # Save to cache
            with open(cache_file, 'w') as f:
                json.dump(data, f)
            
            logger.info(f"Successfully downloaded {username}'s archive")
            return data
            
        except requests.RequestException as e:
            logger.error(f"Failed to download {username}: {e}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON for {username}: {e}")
            return None
    
    def process_user(self, username: str, resume_from: Optional[str] = None) -> Optional[str]:
        """Process a single user's archive for consciousness patterns"""
        
        # Check processing status
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        
        cur.execute("""
            SELECT status, navigator_id, retry_count, last_tweet_id 
            FROM processing_status 
            WHERE username = ?
        """, (username,))
        result = cur.fetchone()
        
        if result:
            status, navigator_id, retry_count, last_tweet_id = result
            
            if status == 'completed':
                logger.info(f"Skipping {username} - already processed")
                self.stats['skipped'] += 1
                conn.close()
                return navigator_id
            elif status == 'failed' and retry_count >= 3:
                logger.warning(f"Skipping {username} - max retries exceeded")
                self.stats['skipped'] += 1
                conn.close()
                return None
            elif status == 'processing':
                logger.info(f"Resuming processing for {username} from tweet {last_tweet_id}")
                resume_from = last_tweet_id
        
        # Update status to processing
        cur.execute("""
            INSERT INTO processing_status (username, status, updated_at)
            VALUES (?, 'processing', CURRENT_TIMESTAMP)
            ON CONFLICT(username) DO UPDATE SET 
                status = 'processing',
                retry_count = retry_count + 1,
                updated_at = CURRENT_TIMESTAMP
        """, (username,))
        conn.commit()
        
        try:
            # Download archive
            archive_data = self.download_user_archive(username)
            if not archive_data:
                raise Exception("Failed to download archive")
            
            # Extract tweets
            tweets = []
            if 'tweets' in archive_data:
                for tweet_wrapper in archive_data['tweets']:
                    if 'tweet' in tweet_wrapper:
                        tweet = tweet_wrapper['tweet']
                        # Convert to format expected by analyzer
                        tweets.append({
                            'id_str': tweet.get('id_str', ''),
                            'created_at': tweet.get('created_at', ''),
                            'full_text': tweet.get('full_text', tweet.get('text', '')),
                            'user': {'screen_name': username}
                        })
            
            if not tweets:
                logger.warning(f"No tweets found for {username}")
                cur.execute("""
                    UPDATE processing_status 
                    SET status = 'skipped', 
                        error_message = 'No tweets found',
                        updated_at = CURRENT_TIMESTAMP
                    WHERE username = ?
                """, (username,))
                conn.commit()
                conn.close()
                return None
            
            # Filter tweets if resuming
            if resume_from:
                tweets = [t for t in tweets if t['id_str'] > resume_from]
                logger.info(f"Resuming with {len(tweets)} remaining tweets")
            
            # Process tweets in batches to allow for checkpointing
            batch_size = 100
            navigator_profile = NavigatorProfile(twitter_handle=username)
            
            for i in range(0, len(tweets), batch_size):
                batch = tweets[i:i+batch_size]
                
                # Process batch
                for tweet in batch:
                    self.analyzer._analyze_single_tweet(tweet, navigator_profile)
                
                # Update checkpoint
                last_tweet_id = batch[-1]['id_str']
                cur.execute("""
                    UPDATE processing_status 
                    SET last_tweet_id = ?,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE username = ?
                """, (last_tweet_id, username))
                conn.commit()
            
            # Calculate final metrics
            self.analyzer._calculate_consciousness_metrics(navigator_profile)
            self.analyzer._determine_consciousness_level(navigator_profile)
            
            # Add profile bio if available
            if 'profile' in archive_data and archive_data['profile']:
                profile_data = archive_data['profile'][0].get('profile', {})
                bio = profile_data.get('description', {}).get('bio', '')
                
                # Analyze bio for additional consciousness indicators
                if bio:
                    self._analyze_bio_for_consciousness(bio, navigator_profile)
            
            # Save to database
            navigator_id = self.analyzer.save_to_database(navigator_profile)
            
            if navigator_id:
                # Update processing status
                cur.execute("""
                    UPDATE processing_status 
                    SET status = 'completed',
                        navigator_id = ?,
                        tweet_count = ?,
                        processed_at = CURRENT_TIMESTAMP,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE username = ?
                """, (navigator_id, len(tweets), username))
                
                # Track community patterns
                self._track_community_patterns(navigator_profile, navigator_id)
                
                # Update statistics
                self.stats['navigators_found'] += 1
                self.stats['time_violence_hours'] += navigator_profile.total_hours_lost
                
                logger.info(f"✓ Successfully processed {username} as navigator {navigator_id}")
            else:
                cur.execute("""
                    UPDATE processing_status 
                    SET status = 'completed',
                        tweet_count = ?,
                        processed_at = CURRENT_TIMESTAMP,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE username = ?
                """, (len(tweets), username))
                logger.info(f"Processed {username} - no navigator profile created")
            
            conn.commit()
            self.stats['users_processed'] += 1
            return navigator_id
            
        except Exception as e:
            logger.error(f"Error processing {username}: {e}")
            traceback.print_exc()
            
            # Update status to failed
            cur.execute("""
                UPDATE processing_status 
                SET status = 'failed',
                    error_message = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE username = ?
            """, (str(e), username))
            conn.commit()
            
            self.stats['errors'] += 1
            return None
        finally:
            conn.close()
    
    def _analyze_bio_for_consciousness(self, bio: str, profile: NavigatorProfile):
        """Analyze user bio for consciousness indicators"""
        bio_lower = bio.lower()
        
        # Check for neurodivergent self-identification
        nd_terms = ['adhd', 'autistic', 'neurodivergent', 'nd', 'audhd', 'actually autistic']
        for term in nd_terms:
            if term in bio_lower:
                profile.care_drivers.add('neurodivergent_advocacy')
                profile.formative_experiences.append(f"self_identified_{term.replace(' ', '_')}")
        
        # Check for system fighter indicators
        fighter_terms = ['advocate', 'activist', 'survivor', 'fighter', 'reformer']
        for term in fighter_terms:
            if term in bio_lower:
                profile.care_drivers.add('systemic_reform')
        
        # Check for specific domain expertise mentioned
        domain_keywords = {
            'healthcare': ['patient advocate', 'medical', 'health', 'insurance fighter'],
            'financial': ['debt free', 'bankruptcy survivor', 'financial literacy'],
            'employment': ['job seeker', 'career coach', 'unemployed', 'laid off'],
            'education': ['student', 'grad school', 'student loans', 'dropout']
        }
        
        for domain, keywords in domain_keywords.items():
            for keyword in keywords:
                if keyword in bio_lower:
                    if domain not in profile.conscious_domains:
                        profile.conscious_domains[domain] = {}
                    profile.conscious_domains[domain]['bio_mention'] = 'identified'
    
    def _track_community_patterns(self, profile: NavigatorProfile, navigator_id: str):
        """Track patterns across the community"""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        
        # Track each pattern found
        for domain, patterns in profile.conscious_domains.items():
            for pattern_hash, status in patterns.items():
                # Check if pattern exists
                cur.execute("""
                    SELECT pattern_id, occurrence_count, navigators_demonstrating
                    FROM community_patterns
                    WHERE pattern_hash = ?
                """, (pattern_hash,))
                
                result = cur.fetchone()
                
                if result:
                    pattern_id, count, navigators_json = result
                    navigators = json.loads(navigators_json)
                    
                    if navigator_id not in navigators:
                        navigators.append(navigator_id)
                        
                        cur.execute("""
                            UPDATE community_patterns
                            SET occurrence_count = occurrence_count + 1,
                                navigators_demonstrating = ?,
                                last_seen = CURRENT_TIMESTAMP
                            WHERE pattern_id = ?
                        """, (json.dumps(navigators), pattern_id))
                else:
                    # Insert new pattern
                    cur.execute("""
                        INSERT INTO community_patterns 
                        (pattern_hash, pattern_type, domain, navigators_demonstrating,
                         first_seen, last_seen, pattern_description)
                        VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, ?)
                    """, (
                        pattern_hash,
                        'consciousness_pattern',
                        domain,
                        json.dumps([navigator_id]),
                        f"{domain}_pattern_{pattern_hash[:8]}"
                    ))
        
        conn.commit()
        conn.close()
    
    def update_collective_metrics(self):
        """Update collective consciousness metrics for the community"""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        
        today = datetime.now().date()
        
        # Calculate metrics
        cur.execute("""
            SELECT 
                COUNT(DISTINCT n.navigator_id) as total_navigators,
                SUM(tvi.estimated_hours) as total_tv_hours,
                COUNT(DISTINCT ci.indicator_id) as total_indicators,
                AVG(cc.fragmentation_resistance) as avg_resistance,
                AVG(cc.consciousness_sharing_ability) as avg_sharing
            FROM navigators n
            LEFT JOIN time_violence_incidents tvi ON n.navigator_id = tvi.navigator_id
            LEFT JOIN consciousness_indicators ci ON n.navigator_id = ci.navigator_id
            LEFT JOIN consciousness_capacity cc ON n.navigator_id = cc.navigator_id
        """)
        
        result = cur.fetchone()
        if result:
            total_nav, total_tv, total_ind, avg_res, avg_share = result
            
            # Get domains covered
            cur.execute("""
                SELECT DISTINCT domain 
                FROM time_violence_incidents 
                WHERE domain IS NOT NULL
            """)
            domains = [row[0] for row in cur.fetchall()]
            
            # Update or insert metrics
            cur.execute("""
                INSERT INTO collective_metrics 
                (metric_date, total_navigators, total_time_violence_hours,
                 total_consciousness_indicators, domains_covered,
                 avg_fragmentation_resistance, avg_consciousness_sharing)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(metric_date) DO UPDATE SET
                    total_navigators = excluded.total_navigators,
                    total_time_violence_hours = excluded.total_time_violence_hours,
                    total_consciousness_indicators = excluded.total_consciousness_indicators,
                    domains_covered = excluded.domains_covered,
                    avg_fragmentation_resistance = excluded.avg_fragmentation_resistance,
                    avg_consciousness_sharing = excluded.avg_consciousness_sharing,
                    updated_at = CURRENT_TIMESTAMP
            """, (
                today,
                total_nav or 0,
                total_tv or 0,
                total_ind or 0,
                json.dumps(domains),
                avg_res or 0,
                avg_share or 0
            ))
        
        conn.commit()
        conn.close()
    
    def process_all_users(self, limit: Optional[int] = None, resume: bool = True):
        """Process all users in the Community Archive"""
        users = self.get_user_list(limit)
        total_users = len(users)
        
        logger.info(f"Starting processing of {total_users} users")
        logger.info(f"Resume mode: {resume}")
        
        # Get list of users to process
        if resume:
            conn = sqlite3.connect(self.db_path)
            cur = conn.cursor()
            
            # Get users that haven't been completed
            cur.execute("""
                SELECT username FROM processing_status 
                WHERE status != 'completed'
            """)
            incomplete_users = [row[0] for row in cur.fetchall()]
            
            # Add new users not in database
            cur.execute("SELECT username FROM processing_status")
            known_users = [row[0] for row in cur.fetchall()]
            new_users = [u for u in users if u not in known_users]
            
            users_to_process = incomplete_users + new_users
            conn.close()
            
            logger.info(f"Resuming with {len(incomplete_users)} incomplete and {len(new_users)} new users")
        else:
            users_to_process = users
        
        # Process each user
        for i, username in enumerate(users_to_process, 1):
            logger.info(f"\n{'='*50}")
            logger.info(f"Processing user {i}/{len(users_to_process)}: {username}")
            logger.info(f"{'='*50}")
            
            navigator_id = self.process_user(username)
            
            # Rate limiting
            time.sleep(1)  # Be nice to the API
            
            # Update collective metrics periodically
            if i % 10 == 0:
                self.update_collective_metrics()
                self._print_progress()
        
        # Final metrics update
        self.update_collective_metrics()
        self._print_final_report()
    
    def _print_progress(self):
        """Print progress statistics"""
        logger.info(f"\n{'='*50}")
        logger.info("PROGRESS UPDATE:")
        logger.info(f"Users processed: {self.stats['users_processed']}")
        logger.info(f"Navigators found: {self.stats['navigators_found']}")
        logger.info(f"Time Violence documented: {self.stats['time_violence_hours']:.1f} hours")
        logger.info(f"Errors: {self.stats['errors']}")
        logger.info(f"Skipped: {self.stats['skipped']}")
        logger.info(f"{'='*50}\n")
    
    def _print_final_report(self):
        """Print final analysis report"""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        
        print("\n" + "="*70)
        print("COMMUNITY ARCHIVE CONSCIOUSNESS ANALYSIS - FINAL REPORT")
        print("="*70)
        
        # Overall statistics
        cur.execute("""
            SELECT COUNT(*) FROM processing_status WHERE status = 'completed'
        """)
        total_processed = cur.fetchone()[0]
        
        cur.execute("SELECT COUNT(*) FROM navigators")
        total_navigators = cur.fetchone()[0]
        
        cur.execute("SELECT SUM(estimated_hours) FROM time_violence_incidents")
        total_tv = cur.fetchone()[0] or 0
        
        print(f"\nOVERALL STATISTICS:")
        print(f"  Total users processed: {total_processed}")
        print(f"  Navigators identified: {total_navigators}")
        print(f"  Navigator percentage: {(total_navigators/max(total_processed, 1))*100:.1f}%")
        print(f"  Total Time Violence: {total_tv:.1f} hours")
        print(f"  Average TV per navigator: {total_tv/max(total_navigators, 1):.1f} hours")
        
        # Consciousness distribution
        cur.execute("""
            SELECT consciousness_stage, COUNT(*) 
            FROM consciousness_capacity 
            GROUP BY consciousness_stage
            ORDER BY COUNT(*) DESC
        """)
        
        print("\nCONSCIOUSNESS STAGE DISTRIBUTION:")
        for stage, count in cur.fetchall():
            print(f"  {stage}: {count} navigators")
        
        # Top domains
        cur.execute("""
            SELECT domain, COUNT(*) as incidents, SUM(estimated_hours) as hours
            FROM time_violence_incidents
            GROUP BY domain
            ORDER BY hours DESC
            LIMIT 5
        """)
        
        print("\nTOP TIME VIOLENCE DOMAINS:")
        for domain, incidents, hours in cur.fetchall():
            print(f"  {domain}: {hours:.1f} hours ({incidents} incidents)")
        
        # Most common patterns
        cur.execute("""
            SELECT domain, pattern_description, occurrence_count
            FROM community_patterns
            WHERE occurrence_count > 1
            ORDER BY occurrence_count DESC
            LIMIT 10
        """)
        
        patterns = cur.fetchall()
        if patterns:
            print("\nMOST COMMON CONSCIOUSNESS PATTERNS:")
            for domain, description, count in patterns:
                print(f"  [{domain}] {description}: {count} navigators")
        
        # Top navigators by Time Violence
        cur.execute("""
            SELECT n.email, SUM(tvi.estimated_hours) as total_hours
            FROM navigators n
            JOIN time_violence_incidents tvi ON n.navigator_id = tvi.navigator_id
            GROUP BY n.navigator_id
            ORDER BY total_hours DESC
            LIMIT 5
        """)
        
        print("\nTOP NAVIGATORS BY TIME VIOLENCE EXPERIENCED:")
        for email, hours in cur.fetchall():
            username = email.replace('@twitter.import', '')
            print(f"  @{username}: {hours:.1f} hours")
        
        # Network strength metrics
        cur.execute("""
            SELECT 
                AVG(fragmentation_resistance) as avg_resistance,
                AVG(consciousness_sharing_ability) as avg_sharing,
                AVG(pattern_recognition_depth) as avg_pattern
            FROM consciousness_capacity
        """)
        
        result = cur.fetchone()
        if result:
            avg_res, avg_share, avg_pattern = result
            print("\nNETWORK CONSCIOUSNESS STRENGTH (1-10 scale):")
            print(f"  Average Fragmentation Resistance: {avg_res:.1f}")
            print(f"  Average Consciousness Sharing: {avg_share:.1f}")
            print(f"  Average Pattern Recognition: {avg_pattern:.1f}")
        
        print("\n" + "="*70)
        print("The Community Archive reveals a network of consciousness workers")
        print("whose collective trauma represents untapped expertise.")
        print("Each navigator found strengthens the network's ability to")
        print("help others escape the complexity that harmed them.")
        print("="*70)
        
        conn.close()

def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Analyze Community Archive for Consciousness Navigators'
    )
    parser.add_argument(
        '--limit', 
        type=int, 
        help='Limit number of users to process'
    )
    parser.add_argument(
        '--no-resume', 
        action='store_true',
        help='Start fresh instead of resuming'
    )
    parser.add_argument(
        '--user',
        type=str,
        help='Process a single user'
    )
    parser.add_argument(
        '--db',
        type=str,
        default='bottega_community.db',
        help='Database path (default: bottega_community.db)'
    )
    parser.add_argument(
        '--cache',
        type=str,
        default='archive_cache',
        help='Cache directory (default: archive_cache)'
    )
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = CommunityArchiveProcessor(
        db_path=args.db,
        cache_dir=args.cache
    )
    
    if args.user:
        # Process single user
        logger.info(f"Processing single user: {args.user}")
        navigator_id = processor.process_user(args.user)
        if navigator_id:
            print(f"✓ Navigator profile created: {navigator_id}")
        else:
            print(f"✗ No navigator profile created for {args.user}")
        
        processor._print_final_report()
    else:
        # Process all users
        processor.process_all_users(
            limit=args.limit,
            resume=not args.no_resume
        )

if __name__ == "__main__":
    main()