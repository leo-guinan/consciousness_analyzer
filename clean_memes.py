#!/usr/bin/env python3
"""
Clean memes and duplicate content from consciousness indicators
"""

import argparse
import logging
from collections import defaultdict

from consciousness_analysis.core.database import DatabaseManager
from consciousness_analysis.utils.meme_filters import (
    detect_duplicate_content,
    is_genuine_consciousness_indicator,
    filter_consciousness_indicators
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def clean_database_memes(db_path: str = "bottega_community.db", dry_run: bool = True):
    """
    Clean memes and duplicates from consciousness indicators

    Args:
        db_path: Path to database
        dry_run: If True, only report what would be deleted without actually deleting
    """
    db_manager = DatabaseManager(db_path)

    # Fetch all consciousness indicators
    indicators = db_manager.execute_query("""
        SELECT
            ci.*,
            n.email
        FROM consciousness_indicators ci
        JOIN navigators n ON ci.navigator_id = n.navigator_id
        ORDER BY ci.content
    """)

    if not indicators:
        logger.info("No consciousness indicators found")
        return

    logger.info(f"Found {len(indicators)} total consciousness indicators")

    # Convert to dict format for processing
    indicator_dicts = []
    for ind in indicators:
        indicator_dicts.append({
            'indicator_id': ind['indicator_id'],
            'navigator_id': ind['navigator_id'],
            'email': ind['email'],
            'content': ind['content'],
            'complexity_domain': ind['complexity_domain'],
            'consciousness_type': ind['consciousness_type'],
            'intensity_score': ind['intensity_score']
        })

    # Detect duplicates
    duplicates = detect_duplicate_content(indicator_dicts)

    logger.info(f"\nFound {len(duplicates)} unique duplicate content patterns")

    # Show top duplicates
    print("\nTop duplicate content (appearing across multiple users):")
    print("-" * 60)
    sorted_dups = sorted(duplicates.items(), key=lambda x: len(x[1]['users']), reverse=True)
    for i, (hash_val, data) in enumerate(sorted_dups[:10], 1):
        print(f"\n{i}. Appears in {len(data['users'])} users:")
        print(f"   Content: {data['content'][:150]}...")
        print(f"   Users: {', '.join(data['users'][:5])}" +
              (f" and {len(data['users'])-5} more" if len(data['users']) > 5 else ""))

    # Filter for genuine indicators
    filtered = filter_consciousness_indicators(indicator_dicts)
    to_remove = set(ind['indicator_id'] for ind in indicator_dicts) - set(ind['indicator_id'] for ind in filtered)

    logger.info(f"\nIdentified {len(to_remove)} indicators to remove")
    logger.info(f"Will keep {len(filtered)} genuine indicators")

    if to_remove:
        # Show examples of what will be removed
        print("\nExamples of indicators to be removed:")
        print("-" * 60)
        examples = [ind for ind in indicator_dicts if ind['indicator_id'] in to_remove][:5]
        for ind in examples:
            print(f"\n[{ind['email']}] ({ind['complexity_domain']}_{ind['consciousness_type']})")
            print(f"  {ind['content'][:200]}...")

        if not dry_run:
            # Actually delete the indicators
            logger.info("\nDeleting meme/duplicate indicators...")

            # Delete in batches
            batch_size = 100
            indicator_list = list(to_remove)

            for i in range(0, len(indicator_list), batch_size):
                batch = indicator_list[i:i+batch_size]
                placeholders = ','.join(['?'] * len(batch))
                query = f"DELETE FROM consciousness_indicators WHERE indicator_id IN ({placeholders})"
                db_manager.execute_write(query, tuple(batch))

            logger.info(f"✓ Deleted {len(to_remove)} meme/duplicate indicators")

            # Update statistics
            remaining = db_manager.execute_query(
                "SELECT COUNT(*) as count FROM consciousness_indicators"
            )[0]['count']
            logger.info(f"✓ {remaining} genuine indicators remain")
        else:
            logger.info("\n(Dry run - no changes made)")
            logger.info("Run with --execute to actually delete these indicators")

    # Check for specific known memes
    william_meme = db_manager.execute_query("""
        SELECT COUNT(*) as count
        FROM consciousness_indicators
        WHERE content LIKE '%william: doctor%'
    """)[0]['count']

    if william_meme > 0:
        logger.info(f"\nFound {william_meme} instances of the 'William doctor' meme")
        if not dry_run:
            db_manager.execute_write("""
                DELETE FROM consciousness_indicators
                WHERE content LIKE '%william: doctor%'
            """)
            logger.info("✓ Deleted William doctor meme instances")


def main():
    parser = argparse.ArgumentParser(
        description='Clean memes and duplicates from consciousness indicators'
    )
    parser.add_argument(
        '--db',
        default='bottega_community.db',
        help='Database path'
    )
    parser.add_argument(
        '--execute',
        action='store_true',
        help='Actually delete (default is dry run)'
    )

    args = parser.parse_args()

    clean_database_memes(
        db_path=args.db,
        dry_run=not args.execute
    )


if __name__ == "__main__":
    main()