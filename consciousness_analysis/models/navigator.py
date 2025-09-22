"""Data models for navigator profiles and consciousness indicators"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Set


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
class TimeViolenceIncident:
    """Represents a Time Violence incident"""
    tweet_id: str
    timestamp: str
    domain: str
    description: str
    estimated_hours: float


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