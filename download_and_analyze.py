#!/usr/bin/env python3
"""
Download archives from Supabase and analyze for consciousness patterns
Can be run from command line or imported into Jupyter notebooks
"""

import argparse
import logging
from pathlib import Path
from typing import Optional, List

from consciousness_analysis.core.database import DatabaseManager
from consciousness_analysis.processors.community_archive import CommunityArchiveProcessor
from consciousness_analysis.utils.supabase_downloader import SupabaseArchiveDownloader
from consciousness_analysis.utils.reporting import generate_community_report

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def download_and_analyze(
    db_path: str = "bottega_community.db",
    cache_dir: str = "archive_cache",
    usernames: Optional[List[str]] = None,
    limit: Optional[int] = None,
    api_key: Optional[str] = None,
    show_progress: bool = True,
    analyze: bool = True
):
    """
    Download user archives from Supabase and optionally analyze them

    Args:
        db_path: Path to SQLite database
        cache_dir: Directory to cache downloads
        usernames: Specific usernames to process, or None for all
        limit: Limit number of users to process
        api_key: Optional Supabase API key
        show_progress: Whether to show progress bars
        analyze: Whether to run consciousness analysis after downloading

    Returns:
        Dictionary of results
    """
    results = {
        'downloaded': [],
        'analyzed': [],
        'failed': []
    }

    # Initialize downloader
    downloader = SupabaseArchiveDownloader(
        cache_dir=cache_dir,
        api_key=api_key
    )

    # Get usernames from Supabase if not provided
    if usernames is None:
        logger.info("Fetching username list from Supabase...")
        usernames = downloader.get_usernames_from_supabase(limit)
        logger.info(f"Found {len(usernames)} usernames to process")
    elif limit:
        usernames = usernames[:limit]

    # Download archives
    logger.info(f"Downloading archives for {len(usernames)} users...")
    downloaded_data = downloader.download_all_users(
        usernames=usernames,
        show_progress=show_progress
    )
    results['downloaded'] = list(downloaded_data.keys())

    # Analyze if requested
    if analyze and downloaded_data:
        logger.info("Starting consciousness analysis...")

        # Initialize database and processor
        db_manager = DatabaseManager(db_path)
        processor = CommunityArchiveProcessor(
            db_manager=db_manager,
            cache_dir=cache_dir
        )

        # Process each downloaded user
        for username in downloaded_data.keys():
            try:
                navigator_id = processor.process_user(username)
                if navigator_id:
                    results['analyzed'].append(username)
                    logger.info(f"✓ Analyzed {username} -> Navigator ID: {navigator_id}")
            except Exception as e:
                results['failed'].append(username)
                logger.error(f"✗ Failed to analyze {username}: {e}")

        # Update collective metrics
        processor.update_collective_metrics()

        # Generate report
        report = generate_community_report(db_manager)
        print(report)

    # Summary
    print("\n" + "="*60)
    print("PROCESSING SUMMARY")
    print("="*60)
    print(f"Downloaded: {len(results['downloaded'])} users")
    if analyze:
        print(f"Analyzed: {len(results['analyzed'])} users")
        print(f"Failed: {len(results['failed'])} users")

    return results


def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(
        description='Download and analyze Twitter archives from Supabase'
    )
    parser.add_argument(
        '--users',
        nargs='+',
        help='Specific usernames to download'
    )
    parser.add_argument(
        '--limit',
        type=int,
        help='Limit number of users to process'
    )
    parser.add_argument(
        '--db',
        default='bottega_community.db',
        help='Database path'
    )
    parser.add_argument(
        '--cache',
        default='archive_cache',
        help='Cache directory'
    )
    parser.add_argument(
        '--api-key',
        help='Supabase API key'
    )
    parser.add_argument(
        '--download-only',
        action='store_true',
        help='Only download, do not analyze'
    )
    parser.add_argument(
        '--no-progress',
        action='store_true',
        help='Disable progress bars'
    )

    args = parser.parse_args()

    download_and_analyze(
        db_path=args.db,
        cache_dir=args.cache,
        usernames=args.users,
        limit=args.limit,
        api_key=args.api_key,
        show_progress=not args.no_progress,
        analyze=not args.download_only
    )


if __name__ == "__main__":
    main()