"""Supabase archive downloader with progress tracking"""

import json
from pathlib import Path
from typing import List, Optional, Dict
import logging
import requests
from supabase import create_client, Client
import os
logger = logging.getLogger(__name__)


class SupabaseArchiveDownloader:
    """Downloads user archives from Supabase storage"""

    ARCHIVE_BASE_URL = "https://fabxmporizzqflnftavs.supabase.co/storage/v1/object/public/archives"

    def __init__(self, cache_dir: str = "archive_cache"):
        """
        Initialize downloader

        Args:
            cache_dir: Directory to cache downloaded archives
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

    def get_usernames_from_supabase(self, limit: Optional[int] = None) -> List[str]:
        """
        Fetch list of available usernames

        Since the Supabase storage list API requires authentication,
        we'll use a known list or check what's been cached.

        Args:
            limit: Optional limit on number of usernames to fetch

        Returns:
            List of usernames available in the archive
        """
        # Check for cached complete list first
        cached_list_path = self.cache_dir / "usernames.json"
        if cached_list_path.exists():
            try:
                with open(cached_list_path, 'r') as f:
                    usernames = json.load(f)
                    logger.info(f"Loaded {len(usernames)} usernames from cache")
                    return usernames[:limit] if limit else usernames
            except Exception as e:
                logger.warning(f"Could not load cached usernames: {e}")

        # Try to discover usernames by checking known archives
        discovered = self._discover_usernames()
        if discovered:
            # Save discovered usernames
            with open(cached_list_path, 'w') as f:
                json.dump(discovered, f)
            logger.info(f"Discovered and cached {len(discovered)} usernames")
            return discovered[:limit] if limit else discovered

        # Fallback to a comprehensive known list
        known_usernames = self._get_known_usernames()

        # Save to cache for next time
        with open(cached_list_path, 'w') as f:
            json.dump(known_usernames, f)

        logger.info(f"Using {len(known_usernames)} known usernames")
        return known_usernames[:limit] if limit else known_usernames

    def _discover_usernames(self) -> List[str]:
        """
        Try to discover usernames by testing known patterns
        or checking what's already been downloaded
        """
        SUPABASE_ANON_KEY=os.getenv('SUPABASE_ANON_KEY')
        SUPABASE_URL="https://fabxmporizzqflnftavs.supabase.co"

        supabase: Client = create_client(
            SUPABASE_URL, SUPABASE_ANON_KEY,
        )

        response = (
            supabase.table("account")
            .select(
                "account_id, username, account_display_name, profile(bio, website, location)"
            )
            # .eq("username", "leo_guinan")
            # .limit(1)
            .execute()
        )

        accounts = response.data


        discovered = []
        account_usernames = [account["username"] for account in accounts]

        return account_usernames

    def _get_known_usernames(self) -> List[str]:
        """
        Return a comprehensive list of known Community Archive usernames
        This can be expanded as more archives are discovered
        """
        return [
            # Original list
            'leo_guinan', 'chrischipmonk', 'princevogel', 'visakanv', 'patio11',
            'swyx', 'levelsio', 'anthilemoon', 'shl', 'balajis',
            'patrick_oshag', 'georgecushen', 'jasonfried', 'dhh',

            # Additional known archives (add more as discovered)
            'naval', 'paulg', 'sama', 'elonmusk', 'pmarca',
            'rabois', 'benedictevans', 'stratechery', 'howardlindzon',
            'fredwilson', 'bhorowitz', 'reid_hoffman', 'garyvee',
            'alexisohanian', 'kevinrose', 'jason', 'chamath',
            'davidlee', 'hunterwalk', 'semil', 'lpolovets',
            'bryce', 'msuster', 'jeffmiller', 'sarahtavel',
            'benedictevans', 'tomtunguz', 'dunkhippo33', 'trengriffin',

            # Tech Twitter personalities
            'dan_abramov', 'addyosmani', 'paul_irish', 'igrigorik',
            'getify', 'wesbos', 'sarah_edo', 'kentcdodds',
            'mjackson', 'ryanflorence', 'sebmarkbage', 'sophiebits',
            'threepointone', 'acdlite', 'trueadm', 'necolas',

            # Crypto/Web3 personalities
            'VitalikButerin', 'SBF_FTX', 'cz_binance', 'brian_armstrong',
            'novogratz', 'APompliano', 'WhalePanda', 'NickSzabo4',
            'aantonop', 'ErikVoorhees', 'rogerkver', 'jimmysong',

            # More startup/tech people
            'tobi', 'dharmesh', 'randfish', 'peterlevels',
            'ajlkn', 'mijustin', 'tylertringas', 'bentossell'
        ]

    def test_username_exists(self, username: str) -> bool:
        """
        Test if a username's archive exists on Supabase

        Args:
            username: Username to test

        Returns:
            True if archive exists, False otherwise
        """
        url = f"{self.ARCHIVE_BASE_URL}/{username}/archive.json"
        try:
            response = requests.head(url, timeout=5)
            return response.status_code == 200
        except:
            return False

    def download_user_data(self, username: str, show_progress: bool = True) -> Optional[Dict]:
        """
        Download user archive data with progress bar

        Args:
            username: Username to download
            show_progress: Whether to show progress bar

        Returns:
            Parsed JSON data or None if download failed
        """
        output_path = self.cache_dir / f"{username}.json"

        # Check if already cached
        if output_path.exists():
            logger.info(f"{output_path} already exists, loading from cache")
            try:
                with open(output_path, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                logger.warning(f"Corrupted cache file for {username}, re-downloading")
                output_path.unlink()

        # Download from Supabase
        url = f"{self.ARCHIVE_BASE_URL}/{username.lower()}/archive.json"
        logger.info(f"Downloading tweet data for: {username}")

        try:
            response = requests.get(url, stream=True, timeout=30)

            # Check if archive exists
            if response.status_code == 404:
                logger.warning(f"Archive not found for {username}")
                return None

            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))

            # Setup progress bar
            progress_bar = None
            if show_progress:
                try:
                    # Try notebook version first
                    from tqdm.notebook import tqdm as tqdm_notebook
                    progress_bar = tqdm_notebook(
                        total=total_size,
                        unit='B',
                        unit_scale=True,
                        desc=f"Downloading {username}"
                    )
                except ImportError:
                    # Fall back to regular tqdm
                    try:
                        from tqdm import tqdm
                        progress_bar = tqdm(
                            total=total_size,
                            unit='B',
                            unit_scale=True,
                            desc=f"Downloading {username}"
                        )
                    except ImportError:
                        # No tqdm available
                        logger.info("Install tqdm for progress bars: pip install tqdm")

            # Download in chunks
            json_data = bytearray()
            for chunk in response.iter_content(1024):  # 1KB chunks
                if chunk:
                    json_data.extend(chunk)
                    if progress_bar:
                        progress_bar.update(len(chunk))

            if progress_bar:
                progress_bar.close()

            # Parse JSON
            data = json.loads(json_data.decode('utf-8', errors='ignore'))

            # Save to cache
            logger.info(f"Writing to cache: {output_path}")
            with open(output_path, 'w') as f:
                json.dump(data, f)

            return data

        except requests.RequestException as e:
            if response.status_code == 404:
                logger.warning(f"Archive does not exist for {username}")
            else:
                logger.error(f"Failed to download {username}: {e}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON for {username}: {e}")
            return None

    def download_all_users(self,
                          usernames: Optional[List[str]] = None,
                          limit: Optional[int] = None,
                          show_progress: bool = True,
                          skip_missing: bool = True) -> Dict[str, Dict]:
        """
        Download archives for multiple users

        Args:
            usernames: List of usernames to download, or None to fetch from known list
            limit: Limit number of users to download
            show_progress: Whether to show progress bars
            skip_missing: Skip usernames that don't have archives

        Returns:
            Dictionary mapping usernames to their archive data
        """
        if usernames is None:
            usernames = self.get_usernames_from_supabase(limit)
        elif limit:
            usernames = usernames[:limit]

        results = {}
        failed = []
        missing = []

        # Overall progress bar
        overall_progress = None
        if show_progress:
            try:
                from tqdm.notebook import tqdm as tqdm_notebook
                overall_progress = tqdm_notebook(
                    total=len(usernames),
                    desc="Overall Progress",
                    unit="users"
                )
            except ImportError:
                try:
                    from tqdm import tqdm
                    overall_progress = tqdm(
                        total=len(usernames),
                        desc="Overall Progress",
                        unit="users"
                    )
                except ImportError:
                    pass

        for username in usernames:
            # Optionally test if archive exists before downloading
            if skip_missing and not (self.cache_dir / f"{username}.json").exists():
                if not self.test_username_exists(username):
                    missing.append(username)
                    logger.info(f"Skipping {username} - archive does not exist")
                    if overall_progress:
                        overall_progress.update(1)
                    continue

            data = self.download_user_data(username, show_progress=show_progress)
            if data:
                results[username] = data
            else:
                failed.append(username)

            if overall_progress:
                overall_progress.update(1)

        if overall_progress:
            overall_progress.close()

        logger.info(f"Successfully downloaded {len(results)}/{len(usernames)} users")
        if missing:
            logger.info(f"Archives not found for: {', '.join(missing[:5])}" +
                       (f" and {len(missing)-5} others" if len(missing) > 5 else ""))
        if failed:
            logger.warning(f"Failed to download: {', '.join(failed)}")

        # Update cached username list with successful downloads
        if results:
            successful_users = list(results.keys())
            cache_path = self.cache_dir / "verified_usernames.json"
            with open(cache_path, 'w') as f:
                json.dump(successful_users, f)

        return results

    def get_cached_archives(self) -> List[str]:
        """
        Get list of usernames that have been cached

        Returns:
            List of usernames with cached archives
        """
        cached = []
        for file in self.cache_dir.glob("*.json"):
            if file.name not in ["usernames.json", "user_list.json", "verified_usernames.json"]:
                cached.append(file.stem)
        return cached

    def discover_new_archives(self, test_usernames: List[str]) -> List[str]:
        """
        Test a list of potential usernames to discover new archives

        Args:
            test_usernames: List of usernames to test

        Returns:
            List of usernames that have archives
        """
        found = []
        logger.info(f"Testing {len(test_usernames)} potential usernames...")

        for username in test_usernames:
            if self.test_username_exists(username):
                found.append(username)
                logger.info(f"✓ Found archive for {username}")

        if found:
            # Update the cached list
            existing = self.get_usernames_from_supabase()
            all_users = list(set(existing + found))

            cache_path = self.cache_dir / "usernames.json"
            with open(cache_path, 'w') as f:
                json.dump(all_users, f)

            logger.info(f"Discovered {len(found)} new archives, total: {len(all_users)}")

        return found