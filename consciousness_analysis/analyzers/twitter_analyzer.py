"""Twitter consciousness analysis module"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Optional
import uuid
import traceback

from ..core.database import DatabaseManager
from ..models.navigator import NavigatorProfile, ConsciousnessIndicator
from ..config.patterns import DOMAIN_PATTERNS, NEURODIVERGENT_PATTERNS
from ..utils.text_analysis import (
    extract_time_amount, classify_consciousness_type,
    calculate_intensity, generate_situation_hash, analyze_sentiment
)
from ..utils.meme_filters import is_genuine_consciousness_indicator

logger = logging.getLogger(__name__)


class TwitterConsciousnessAnalyzer:
    """Analyzes Twitter archives for consciousness patterns"""

    def __init__(self, db_manager: DatabaseManager):
        """Initialize analyzer with database manager"""
        self.db_manager = db_manager
        self.navigator_profile = None
        self.consciousness_indicators = []

    def load_twitter_archive(self, archive_path: str) -> List[Dict]:
        """Load tweets from Twitter archive file"""
        tweets = []

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
            logger.error(f"Error loading archive: {e}")

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
        for domain, patterns in DOMAIN_PATTERNS.items():

            # Count keyword matches
            keyword_matches = sum(1 for kw in patterns['keywords'] if kw in text)

            if keyword_matches > 0:

                # Check for Time Violence
                tv_indicators = sum(1 for ind in patterns['time_violence_indicators'] if ind in text)
                if tv_indicators > 0:
                    time_wasted = extract_time_amount(text)
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
                    # Validate it's genuine consciousness, not a meme
                    if not is_genuine_consciousness_indicator(text, domain):
                        logger.debug(f"Filtered out likely meme/copypasta: {text[:100]}...")
                        continue

                    # This shows pattern recognition / consciousness development
                    situation_hash = generate_situation_hash(domain, text)

                    indicator = ConsciousnessIndicator(
                        tweet_id=tweet_id,
                        timestamp=created_at,
                        content=text,
                        complexity_domain=domain,
                        consciousness_type=classify_consciousness_type(text),
                        intensity_score=calculate_intensity(text),
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
                sentiment = analyze_sentiment(text)
                if sentiment['compound'] < -0.5 or sentiment['compound'] > 0.5:
                    profile.care_drivers.add(f"{domain}_reform")

        # Check for neurodivergent patterns
        nd_matches = sum(1 for pattern in NEURODIVERGENT_PATTERNS if pattern in text)
        if nd_matches > 0:
            profile.care_drivers.add('neurodivergent_advocacy')

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

        # Consciousness sharing ability - based on consciousness demonstrations
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

    def save_to_database(self, profile: NavigatorProfile) -> Optional[str]:
        """Save navigator profile to database"""

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