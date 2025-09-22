"""Community Archive processing module"""

import json
import logging
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import requests

from ..core.database import DatabaseManager
from ..models.navigator import NavigatorProfile
from ..analyzers.twitter_analyzer import TwitterConsciousnessAnalyzer
from ..config.patterns import BIO_KEYWORDS

logger = logging.getLogger(__name__)


class CommunityArchiveProcessor:
    """Processes Community Archive data to discover consciousness navigators"""

    ARCHIVE_BASE_URL = "https://fabxmporizzqflnftavs.supabase.co/storage/v1/object/public/archives"

    def __init__(self, db_manager: DatabaseManager, cache_dir: str = "archive_cache"):
        """Initialize processor with database manager and cache directory"""
        self.db_manager = db_manager
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        # Initialize consciousness analyzer
        self.analyzer = TwitterConsciousnessAnalyzer(db_manager)

        # Track processing statistics
        self.stats = {
            'users_processed': 0,
            'navigators_found': 0,
            'time_violence_hours': 0,
            'errors': 0,
            'skipped': 0
        }

    def get_user_list(self, limit: Optional[int] = None) -> List[str]:
        """Get list of users from Community Archive"""
        # Use the new Supabase downloader if available
        try:
            from ..utils.supabase_downloader import SupabaseArchiveDownloader
            downloader = SupabaseArchiveDownloader(cache_dir=str(self.cache_dir))
            users = downloader.get_usernames_from_supabase(limit)
            logger.info(f"Fetched {len(users)} users from Supabase")
            return users
        except ImportError:
            pass

        # Fallback to cache
        user_list_file = self.cache_dir / "user_list.json"

        if user_list_file.exists():
            with open(user_list_file, 'r') as f:
                users = json.load(f)
                logger.info(f"Loaded {len(users)} users from cache")
        else:
            # Fallback to known users
            users = [
                'leo_guinan', 'chrischipmonk', 'princevogel', 'visakanv', 'patio11',
                'swyx', 'levelsio', 'anthilemoon', 'shl', 'balajis',
                'patrick_oshag', 'georgecushen', 'jasonfried', 'dhh'
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
        url = f"{self.ARCHIVE_BASE_URL}/{username.lower()}/archive.json"
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
        result = self.db_manager.execute_query("""
            SELECT status, navigator_id, retry_count, last_tweet_id
            FROM processing_status
            WHERE username = ?
        """, (username,))

        if result:
            row = result[0]
            status, navigator_id, retry_count, last_tweet_id = row['status'], row['navigator_id'], row['retry_count'], row['last_tweet_id']

            if status == 'completed':
                logger.info(f"Skipping {username} - already processed")
                self.stats['skipped'] += 1
                return navigator_id
            elif status == 'failed' and retry_count >= 3:
                logger.warning(f"Skipping {username} - max retries exceeded")
                self.stats['skipped'] += 1
                return None
            elif status == 'processing':
                logger.info(f"Resuming processing for {username} from tweet {last_tweet_id}")
                resume_from = last_tweet_id

        # Update status to processing
        self.db_manager.execute_write("""
            INSERT INTO processing_status (username, status, updated_at)
            VALUES (?, 'processing', CURRENT_TIMESTAMP)
            ON CONFLICT(username) DO UPDATE SET
                status = 'processing',
                retry_count = retry_count + 1,
                updated_at = CURRENT_TIMESTAMP
        """, (username,))

        try:
            # Download archive
            archive_data = self.download_user_archive(username)
            if not archive_data:
                raise Exception("Failed to download archive")

            # Extract tweets
            tweets = self._extract_tweets(archive_data, username)

            if not tweets:
                logger.warning(f"No tweets found for {username}")
                self._update_processing_status(username, 'skipped', error_message='No tweets found')
                return None

            # Filter tweets if resuming
            if resume_from:
                tweets = [t for t in tweets if t['id_str'] > resume_from]
                logger.info(f"Resuming with {len(tweets)} remaining tweets")

            # Process tweets
            navigator_profile = self._process_tweets(tweets, username, archive_data)

            # Save to database
            navigator_id = self.analyzer.save_to_database(navigator_profile)

            if navigator_id:
                # Update processing status
                self._update_processing_status(
                    username, 'completed',
                    navigator_id=navigator_id,
                    tweet_count=len(tweets)
                )

                # Track community patterns
                self._track_community_patterns(navigator_profile, navigator_id)

                # Update statistics
                self.stats['navigators_found'] += 1
                self.stats['time_violence_hours'] += navigator_profile.total_hours_lost

                logger.info(f"✓ Successfully processed {username} as navigator {navigator_id}")
            else:
                self._update_processing_status(username, 'completed', tweet_count=len(tweets))
                logger.info(f"Processed {username} - no navigator profile created")

            self.stats['users_processed'] += 1
            return navigator_id

        except Exception as e:
            logger.error(f"Error processing {username}: {e}")
            traceback.print_exc()

            # Update status to failed
            self._update_processing_status(username, 'failed', error_message=str(e))
            self.stats['errors'] += 1
            return None

    def _extract_tweets(self, archive_data: Dict, username: str) -> List[Dict]:
        """Extract tweets from archive data"""
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
        return tweets

    def _process_tweets(self, tweets: List[Dict], username: str, archive_data: Dict) -> NavigatorProfile:
        """Process tweets in batches"""
        batch_size = 100
        navigator_profile = NavigatorProfile(twitter_handle=username)

        for i in range(0, len(tweets), batch_size):
            batch = tweets[i:i+batch_size]

            # Process batch
            for tweet in batch:
                self.analyzer._analyze_single_tweet(tweet, navigator_profile)

            # Update checkpoint
            if batch:
                last_tweet_id = batch[-1]['id_str']
                self.db_manager.execute_write("""
                    UPDATE processing_status
                    SET last_tweet_id = ?,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE username = ?
                """, (last_tweet_id, username))

        # Calculate final metrics
        self.analyzer._calculate_consciousness_metrics(navigator_profile)
        self.analyzer._determine_consciousness_level(navigator_profile)

        # Add profile bio if available
        if 'profile' in archive_data and archive_data['profile']:
            profile_data = archive_data['profile'][0].get('profile', {})
            bio = profile_data.get('description', {}).get('bio', '')

            if bio:
                self._analyze_bio_for_consciousness(bio, navigator_profile)

        return navigator_profile

    def _analyze_bio_for_consciousness(self, bio: str, profile: NavigatorProfile):
        """Analyze user bio for consciousness indicators"""
        bio_lower = bio.lower()

        # Check for neurodivergent self-identification
        for term in BIO_KEYWORDS['neurodivergent_terms']:
            if term in bio_lower:
                profile.care_drivers.add('neurodivergent_advocacy')
                profile.formative_experiences.append(f"self_identified_{term.replace(' ', '_')}")

        # Check for system fighter indicators
        for term in BIO_KEYWORDS['fighter_terms']:
            if term in bio_lower:
                profile.care_drivers.add('systemic_reform')

        # Check for specific domain expertise mentioned
        for domain, keywords in BIO_KEYWORDS['domain_expertise'].items():
            for keyword in keywords:
                if keyword in bio_lower:
                    if domain not in profile.conscious_domains:
                        profile.conscious_domains[domain] = {}
                    profile.conscious_domains[domain]['bio_mention'] = 'identified'

    def _track_community_patterns(self, profile: NavigatorProfile, navigator_id: str):
        """Track patterns across the community"""
        for domain, patterns in profile.conscious_domains.items():
            for pattern_hash, status in patterns.items():
                # Check if pattern exists
                result = self.db_manager.execute_query("""
                    SELECT pattern_id, occurrence_count, navigators_demonstrating
                    FROM community_patterns
                    WHERE pattern_hash = ?
                """, (pattern_hash,))

                if result:
                    row = result[0]
                    pattern_id = row['pattern_id']
                    navigators = json.loads(row['navigators_demonstrating'])

                    if navigator_id not in navigators:
                        navigators.append(navigator_id)

                        self.db_manager.execute_write("""
                            UPDATE community_patterns
                            SET occurrence_count = occurrence_count + 1,
                                navigators_demonstrating = ?,
                                last_seen = CURRENT_TIMESTAMP
                            WHERE pattern_id = ?
                        """, (json.dumps(navigators), pattern_id))
                else:
                    # Insert new pattern
                    self.db_manager.execute_write("""
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

    def _update_processing_status(self, username: str, status: str,
                                 navigator_id: Optional[str] = None,
                                 tweet_count: Optional[int] = None,
                                 error_message: Optional[str] = None):
        """Update processing status for a user"""
        params = [status]
        query_parts = ["UPDATE processing_status SET status = ?"]

        if navigator_id:
            query_parts.append("navigator_id = ?")
            params.append(navigator_id)

        if tweet_count is not None:
            query_parts.append("tweet_count = ?")
            params.append(tweet_count)

        if error_message:
            query_parts.append("error_message = ?")
            params.append(error_message)

        if status == 'completed':
            query_parts.append("processed_at = CURRENT_TIMESTAMP")

        query_parts.append("updated_at = CURRENT_TIMESTAMP")
        query = ", ".join(query_parts) + " WHERE username = ?"
        params.append(username)

        self.db_manager.execute_write(query, tuple(params))

    def update_collective_metrics(self):
        """Update collective consciousness metrics for the community"""
        today = datetime.now().date()

        # Calculate metrics
        result = self.db_manager.execute_query("""
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

        if result:
            row = result[0]
            total_nav = row['total_navigators'] or 0
            total_tv = row['total_tv_hours'] or 0
            total_ind = row['total_indicators'] or 0
            avg_res = row['avg_resistance'] or 0
            avg_share = row['avg_sharing'] or 0

            # Get domains covered
            domains_result = self.db_manager.execute_query("""
                SELECT DISTINCT domain
                FROM time_violence_incidents
                WHERE domain IS NOT NULL
            """)
            domains = [row['domain'] for row in domains_result]

            # Update or insert metrics
            self.db_manager.execute_write("""
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
                total_nav,
                total_tv,
                total_ind,
                json.dumps(domains),
                avg_res,
                avg_share
            ))

    def process_all_users(self, limit: Optional[int] = None, resume: bool = True):
        """Process all users in the Community Archive"""
        users = self.get_user_list(limit)
        total_users = len(users)

        logger.info(f"Starting processing of {total_users} users")
        logger.info(f"Resume mode: {resume}")

        # Get list of users to process
        if resume:
            # Get users that haven't been completed
            incomplete_result = self.db_manager.execute_query("""
                SELECT username FROM processing_status
                WHERE status != 'completed'
            """)
            incomplete_users = [row['username'] for row in incomplete_result]

            # Get all known users
            known_result = self.db_manager.execute_query("SELECT username FROM processing_status")
            known_users = [row['username'] for row in known_result]

            # Add new users not in database
            new_users = [u for u in users if u not in known_users]

            users_to_process = incomplete_users + new_users
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