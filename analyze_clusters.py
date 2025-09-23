#!/usr/bin/env python3
"""
Analyze consciousness clusters and export data for further analysis
"""

import argparse
import logging
import json
from pathlib import Path

from consciousness_analysis.core.database import DatabaseManager
from consciousness_analysis.analyzers.consciousness_clustering import ConsciousnessClusterAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def analyze_clusters(
    db_path: str = "bottega_community.db",
    output_dir: str = "consciousness_clusters",
    min_shared_patterns: int = 3,
    export_json: bool = True
):
    """
    Analyze consciousness clusters and export results

    Args:
        db_path: Path to SQLite database
        output_dir: Directory for CSV outputs
        min_shared_patterns: Minimum shared patterns for community detection
        export_json: Whether to export JSON summary
    """
    logger.info("="*60)
    logger.info("CONSCIOUSNESS CLUSTER ANALYSIS")
    logger.info("="*60)

    # Initialize database and analyzer
    db_manager = DatabaseManager(db_path)
    analyzer = ConsciousnessClusterAnalyzer(db_manager, output_dir)

    # Run cluster analysis
    logger.info("\n1. Analyzing consciousness clusters...")
    cluster_results = analyzer.analyze_consciousness_clusters()

    if not cluster_results:
        logger.warning("No consciousness data found to analyze")
        return

    # Print cluster statistics
    if 'statistics' in cluster_results:
        stats = cluster_results['statistics']
        print("\nCLUSTER STATISTICS:")
        print("-" * 40)
        print(f"Total clusters: {stats.get('total_clusters', 0)}")
        print(f"Total indicators: {stats.get('total_indicators', 0)}")
        print(f"Avg indicators per cluster: {stats.get('avg_indicators_per_cluster', 0):.1f}")
        print(f"Avg navigators per cluster: {stats.get('avg_navigators_per_cluster', 0):.1f}")

        if 'domain_distribution' in stats:
            print("\nDOMAIN DISTRIBUTION:")
            for domain, count in sorted(stats['domain_distribution'].items(), key=lambda x: x[1], reverse=True):
                print(f"  {domain}: {count} indicators")

        if 'type_distribution' in stats:
            print("\nCONSCIOUSNESS TYPE DISTRIBUTION:")
            for cons_type, count in sorted(stats['type_distribution'].items(), key=lambda x: x[1], reverse=True):
                print(f"  {cons_type}: {count} indicators")

    # Print overlap statistics
    if 'overlaps' in cluster_results and 'statistics' in cluster_results['overlaps']:
        overlap_stats = cluster_results['overlaps']['statistics']
        print("\nOVERLAP STATISTICS:")
        print("-" * 40)
        print(f"Total navigators: {overlap_stats['total_navigators']}")
        print(f"Multi-cluster navigators: {overlap_stats['multi_cluster_navigators']}")
        print(f"Cross-domain navigators: {overlap_stats['cross_domain_navigators']}")
        print(f"Avg clusters per navigator: {overlap_stats['avg_clusters_per_navigator']:.1f}")
        print(f"Max clusters per navigator: {overlap_stats['max_clusters_per_navigator']}")

    # Export consciousness matrix
    logger.info("\n2. Exporting consciousness matrix...")
    matrix_file = analyzer.export_consciousness_matrix()
    if matrix_file:
        print(f"\nExported consciousness matrix to: {matrix_file}")

    # Find consciousness communities
    logger.info("\n3. Finding consciousness communities...")
    communities = analyzer.find_consciousness_communities(min_shared_patterns)

    if communities and 'communities' in communities:
        print(f"\nCONSCIOUSNESS COMMUNITIES:")
        print("-" * 40)
        print(f"Found {communities['total_communities']} communities")

        if communities['largest_community']:
            largest = communities['largest_community']
            print(f"Largest community: {len(largest['members'])} members")
            print(f"Shared patterns: {len(largest['shared_patterns'])}")

    # Print common patterns
    if 'common_patterns' in cluster_results:
        patterns = cluster_results['common_patterns']
        if 'overall' in patterns and patterns['overall']:
            print("\nMOST COMMON PATTERNS (Top 10):")
            print("-" * 40)
            for pattern, count in list(patterns['overall'].items())[:10]:
                print(f"  '{pattern}': {count} occurrences")

    # Export JSON summary if requested
    if export_json:
        json_file = Path(output_dir) / "cluster_analysis_summary.json"
        summary = {
            'statistics': cluster_results.get('statistics', {}),
            'overlap_statistics': cluster_results.get('overlaps', {}).get('statistics', {}),
            'common_patterns': cluster_results.get('common_patterns', {}),
            'community_count': communities.get('total_communities', 0) if communities else 0,
            'cluster_count': len(cluster_results.get('clusters', {}))
        }

        with open(json_file, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"\nExported JSON summary to: {json_file}")

    # Print file locations
    print("\n" + "="*60)
    print("EXPORTED FILES:")
    print("="*60)
    output_path = Path(output_dir)
    if output_path.exists():
        files = list(output_path.glob("*.csv"))
        for f in sorted(files):
            print(f"  - {f.name}")

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"All CSV files exported to: {output_dir}/")
    print("\nYou can now:")
    print("1. Open the CSV files in Excel or Google Sheets")
    print("2. Analyze consciousness_matrix.csv to see navigator patterns")
    print("3. Review individual cluster files (*_tweets.csv)")
    print("4. Examine navigator_overlaps.csv for cross-domain analysis")
    print("5. Study consciousness_communities.csv for navigator groups")


def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(
        description='Analyze consciousness clusters and patterns'
    )
    parser.add_argument(
        '--db',
        default='bottega_community.db',
        help='Database path (default: bottega_community.db)'
    )
    parser.add_argument(
        '--output',
        default='consciousness_clusters',
        help='Output directory for CSV files (default: consciousness_clusters)'
    )
    parser.add_argument(
        '--min-shared',
        type=int,
        default=3,
        help='Minimum shared patterns for community detection (default: 3)'
    )
    parser.add_argument(
        '--no-json',
        action='store_true',
        help='Skip JSON summary export'
    )

    args = parser.parse_args()

    analyze_clusters(
        db_path=args.db,
        output_dir=args.output,
        min_shared_patterns=args.min_shared,
        export_json=not args.no_json
    )


if __name__ == "__main__":
    main()