#!/usr/bin/env python3
"""
Community Archive Consciousness Navigator Discovery Pipeline
Main entry point for processing Twitter archives to identify consciousness navigators
"""

import argparse
import logging
from pathlib import Path

from consciousness_analysis.core.database import DatabaseManager
from consciousness_analysis.processors.community_archive import CommunityArchiveProcessor
from consciousness_analysis.utils.reporting import generate_community_report

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


def main():
    """Main execution function"""

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

    # Initialize database manager
    db_manager = DatabaseManager(args.db)

    # Initialize processor
    processor = CommunityArchiveProcessor(
        db_manager=db_manager,
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

        # Generate and print report
        report = generate_community_report(db_manager)
        print(report)
    else:
        # Process all users
        processor.process_all_users(
            limit=args.limit,
            resume=not args.no_resume
        )

        # Generate and print final report
        report = generate_community_report(db_manager)
        print(report)


if __name__ == "__main__":
    main()