"""Consciousness clustering and overlap analysis module"""

import csv
import json
import logging
from collections import defaultdict, Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
import numpy as np

from ..core.database import DatabaseManager
from ..config.patterns import DOMAIN_PATTERNS

logger = logging.getLogger(__name__)


class ConsciousnessClusterAnalyzer:
    """Analyzes consciousness patterns to find clusters and overlaps"""

    def __init__(self, db_manager: DatabaseManager, output_dir: str = "consciousness_clusters"):
        """
        Initialize cluster analyzer

        Args:
            db_manager: Database manager instance
            output_dir: Directory for CSV outputs
        """
        self.db_manager = db_manager
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def analyze_consciousness_clusters(self) -> Dict:
        """
        Analyze all consciousness indicators to find clusters and patterns

        Returns:
            Dictionary with cluster analysis results
        """
        logger.info("Starting consciousness cluster analysis...")

        # Fetch all consciousness indicators
        indicators = self.db_manager.execute_query("""
            SELECT
                ci.*,
                n.email,
                cc.consciousness_stage,
                cc.complexity_scale
            FROM consciousness_indicators ci
            JOIN navigators n ON ci.navigator_id = n.navigator_id
            JOIN consciousness_capacity cc ON ci.navigator_id = cc.navigator_id
            ORDER BY ci.complexity_domain, ci.consciousness_type
        """)

        if not indicators:
            logger.warning("No consciousness indicators found")
            return {}

        # Group by domain and type
        clusters = self._group_indicators_by_pattern(indicators)

        # Analyze overlaps
        overlaps = self._analyze_navigator_overlaps(indicators)

        # Find common patterns
        common_patterns = self._identify_common_patterns(indicators)

        # Calculate statistics
        stats = self._calculate_cluster_statistics(clusters, overlaps)

        # Export to CSV files
        self._export_clusters_to_csv(clusters, overlaps)

        return {
            'clusters': clusters,
            'overlaps': overlaps,
            'common_patterns': common_patterns,
            'statistics': stats
        }

    def _group_indicators_by_pattern(self, indicators: List) -> Dict:
        """Group indicators by domain and consciousness type"""
        clusters = defaultdict(lambda: defaultdict(list))

        for ind in indicators:
            domain = ind['complexity_domain']
            cons_type = ind['consciousness_type']

            cluster_key = f"{domain}_{cons_type}"
            clusters[cluster_key]['indicators'].append({
                'navigator_id': ind['navigator_id'],
                'email': ind['email'],
                'tweet_id': ind['tweet_id'],
                'timestamp': ind['timestamp'],
                'content': ind['content'],
                'intensity': ind['intensity_score'],
                'stage': ind['consciousness_stage'],
                'scale': ind['complexity_scale']
            })

            # Track unique navigators
            if 'navigators' not in clusters[cluster_key]:
                clusters[cluster_key]['navigators'] = set()
            clusters[cluster_key]['navigators'].add(ind['navigator_id'])

            # Metadata
            clusters[cluster_key]['domain'] = domain
            clusters[cluster_key]['type'] = cons_type
            clusters[cluster_key]['count'] = len(clusters[cluster_key]['indicators'])

        # Convert sets to lists for JSON serialization
        for key in clusters:
            clusters[key]['navigators'] = list(clusters[key]['navigators'])
            clusters[key]['navigator_count'] = len(clusters[key]['navigators'])

        return dict(clusters)

    def _analyze_navigator_overlaps(self, indicators: List) -> Dict:
        """Analyze which navigators appear in multiple consciousness clusters"""
        navigator_clusters = defaultdict(set)
        navigator_domains = defaultdict(set)
        navigator_types = defaultdict(set)

        for ind in indicators:
            nav_id = ind['navigator_id']
            domain = ind['complexity_domain']
            cons_type = ind['consciousness_type']
            cluster_key = f"{domain}_{cons_type}"

            navigator_clusters[nav_id].add(cluster_key)
            navigator_domains[nav_id].add(domain)
            navigator_types[nav_id].add(cons_type)

        # Find navigators with multiple clusters
        multi_cluster_navigators = {
            nav_id: {
                'clusters': list(clusters),
                'domains': list(navigator_domains[nav_id]),
                'types': list(navigator_types[nav_id]),
                'cluster_count': len(clusters),
                'domain_count': len(navigator_domains[nav_id]),
                'type_count': len(navigator_types[nav_id])
            }
            for nav_id, clusters in navigator_clusters.items()
            if len(clusters) > 1
        }

        # Calculate overlap statistics
        overlap_stats = {
            'total_navigators': len(navigator_clusters),
            'multi_cluster_navigators': len(multi_cluster_navigators),
            'avg_clusters_per_navigator': np.mean([len(c) for c in navigator_clusters.values()]),
            'max_clusters_per_navigator': max([len(c) for c in navigator_clusters.values()]) if navigator_clusters else 0,
            'cross_domain_navigators': len([n for n, d in navigator_domains.items() if len(d) > 1])
        }

        return {
            'navigator_overlaps': multi_cluster_navigators,
            'statistics': overlap_stats
        }

    def _identify_common_patterns(self, indicators: List) -> Dict:
        """Identify the most common consciousness patterns"""
        # Extract pattern keywords from content
        pattern_counter = Counter()
        domain_pattern_counter = defaultdict(Counter)
        type_pattern_counter = defaultdict(Counter)

        for ind in indicators:
            content = ind['content'].lower()
            domain = ind['complexity_domain']
            cons_type = ind['consciousness_type']

            # Look for specific pattern keywords
            if domain in DOMAIN_PATTERNS:
                for keyword in DOMAIN_PATTERNS[domain].get('consciousness_indicators', []):
                    if keyword in content:
                        pattern_counter[keyword] += 1
                        domain_pattern_counter[domain][keyword] += 1
                        type_pattern_counter[cons_type][keyword] += 1

        # Get top patterns
        top_patterns = {
            'overall': dict(pattern_counter.most_common(20)),
            'by_domain': {
                domain: dict(counter.most_common(10))
                for domain, counter in domain_pattern_counter.items()
            },
            'by_type': {
                cons_type: dict(counter.most_common(10))
                for cons_type, counter in type_pattern_counter.items()
            }
        }

        return top_patterns

    def _calculate_cluster_statistics(self, clusters: Dict, overlaps: Dict) -> Dict:
        """Calculate detailed statistics for clusters"""
        if not clusters:
            return {}

        stats = {
            'total_clusters': len(clusters),
            'total_indicators': sum(c['count'] for c in clusters.values()),
            'avg_indicators_per_cluster': np.mean([c['count'] for c in clusters.values()]),
            'avg_navigators_per_cluster': np.mean([c['navigator_count'] for c in clusters.values()]),
            'largest_cluster': max(clusters.items(), key=lambda x: x[1]['count'])[0] if clusters else None,
            'most_navigators_cluster': max(clusters.items(), key=lambda x: x[1]['navigator_count'])[0] if clusters else None
        }

        # Domain distribution
        domain_counts = Counter()
        for cluster in clusters.values():
            domain_counts[cluster['domain']] += cluster['count']
        stats['domain_distribution'] = dict(domain_counts)

        # Type distribution
        type_counts = Counter()
        for cluster in clusters.values():
            type_counts[cluster['type']] += cluster['count']
        stats['type_distribution'] = dict(type_counts)

        # Add overlap statistics
        if 'statistics' in overlaps:
            stats['overlap_stats'] = overlaps['statistics']

        return stats

    def _export_clusters_to_csv(self, clusters: Dict, overlaps: Dict):
        """Export cluster data to CSV files for further analysis"""
        logger.info(f"Exporting cluster data to {self.output_dir}")

        # Export main cluster file
        cluster_file = self.output_dir / "consciousness_clusters.csv"
        with open(cluster_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'cluster_id', 'domain', 'consciousness_type',
                'indicator_count', 'navigator_count', 'navigator_ids'
            ])

            for cluster_id, data in clusters.items():
                writer.writerow([
                    cluster_id,
                    data['domain'],
                    data['type'],
                    data['count'],
                    data['navigator_count'],
                    '|'.join(data['navigators'])
                ])

        logger.info(f"Exported cluster summary to {cluster_file}")

        # Export detailed indicators for each cluster
        for cluster_id, data in clusters.items():
            if not data['indicators']:
                continue

            filename = f"{cluster_id}_tweets.csv"
            filepath = self.output_dir / filename

            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'navigator_id', 'email', 'tweet_id', 'timestamp',
                    'intensity', 'consciousness_stage', 'complexity_scale', 'content'
                ])

                for ind in data['indicators']:
                    writer.writerow([
                        ind['navigator_id'],
                        ind['email'],
                        ind['tweet_id'],
                        ind['timestamp'],
                        ind['intensity'],
                        ind['stage'],
                        ind['scale'],
                        ind['content']
                    ])

            logger.info(f"Exported {data['count']} tweets to {filename}")

        # Export navigator overlap analysis
        if 'navigator_overlaps' in overlaps:
            overlap_file = self.output_dir / "navigator_overlaps.csv"
            with open(overlap_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'navigator_id', 'cluster_count', 'domain_count',
                    'type_count', 'clusters', 'domains', 'types'
                ])

                for nav_id, data in overlaps['navigator_overlaps'].items():
                    writer.writerow([
                        nav_id,
                        data['cluster_count'],
                        data['domain_count'],
                        data['type_count'],
                        '|'.join(data['clusters']),
                        '|'.join(data['domains']),
                        '|'.join(data['types'])
                    ])

            logger.info(f"Exported navigator overlaps to {overlap_file}")

    def export_consciousness_matrix(self) -> str:
        """Export a matrix showing navigator participation across consciousness types"""
        # Get all navigators and consciousness types
        result = self.db_manager.execute_query("""
            SELECT DISTINCT
                n.navigator_id,
                n.email,
                ci.complexity_domain,
                ci.consciousness_type,
                COUNT(*) as indicator_count
            FROM navigators n
            JOIN consciousness_indicators ci ON n.navigator_id = ci.navigator_id
            GROUP BY n.navigator_id, ci.complexity_domain, ci.consciousness_type
        """)

        if not result:
            logger.warning("No data for consciousness matrix")
            return None

        # Build matrix structure
        navigators = {}
        all_domains = set()
        all_types = set()

        for row in result:
            nav_id = row['navigator_id']
            email = row['email'].replace('@twitter.import', '')
            domain = row['complexity_domain']
            cons_type = row['consciousness_type']
            count = row['indicator_count']

            if nav_id not in navigators:
                navigators[nav_id] = {'email': email, 'patterns': {}}

            pattern_key = f"{domain}_{cons_type}"
            navigators[nav_id]['patterns'][pattern_key] = count

            all_domains.add(domain)
            all_types.add(cons_type)

        # Create all possible combinations
        all_patterns = []
        for domain in sorted(all_domains):
            for cons_type in sorted(all_types):
                all_patterns.append(f"{domain}_{cons_type}")

        # Export matrix to CSV
        matrix_file = self.output_dir / "consciousness_matrix.csv"
        with open(matrix_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            # Header
            header = ['navigator_email'] + all_patterns
            writer.writerow(header)

            # Data rows
            for nav_id, data in navigators.items():
                row = [data['email']]
                for pattern in all_patterns:
                    row.append(data['patterns'].get(pattern, 0))
                writer.writerow(row)

        logger.info(f"Exported consciousness matrix to {matrix_file}")
        return str(matrix_file)

    def find_consciousness_communities(self, min_shared_patterns: int = 3) -> Dict:
        """
        Find communities of navigators with similar consciousness patterns

        Args:
            min_shared_patterns: Minimum number of shared patterns to form a community

        Returns:
            Dictionary of communities and their members
        """
        # Get navigator patterns
        result = self.db_manager.execute_query("""
            SELECT
                n.navigator_id,
                n.email,
                GROUP_CONCAT(ci.complexity_domain || '_' || ci.consciousness_type) as patterns
            FROM navigators n
            JOIN consciousness_indicators ci ON n.navigator_id = ci.navigator_id
            GROUP BY n.navigator_id
        """)

        if not result:
            return {}

        # Build pattern sets for each navigator
        navigator_patterns = {}
        for row in result:
            patterns = set(row['patterns'].split(',')) if row['patterns'] else set()
            navigator_patterns[row['navigator_id']] = {
                'email': row['email'],
                'patterns': patterns
            }

        # Find communities based on shared patterns
        communities = []
        processed = set()

        for nav1_id, nav1_data in navigator_patterns.items():
            if nav1_id in processed:
                continue

            community = {
                'members': [nav1_id],
                'emails': [nav1_data['email']],
                'shared_patterns': nav1_data['patterns'].copy()
            }

            for nav2_id, nav2_data in navigator_patterns.items():
                if nav2_id == nav1_id or nav2_id in processed:
                    continue

                shared = nav1_data['patterns'] & nav2_data['patterns']
                if len(shared) >= min_shared_patterns:
                    community['members'].append(nav2_id)
                    community['emails'].append(nav2_data['email'])
                    community['shared_patterns'] &= nav2_data['patterns']

            if len(community['members']) > 1:
                communities.append(community)
                processed.update(community['members'])

        # Export communities
        if communities:
            community_file = self.output_dir / "consciousness_communities.csv"
            with open(community_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['community_id', 'member_count', 'shared_pattern_count', 'members', 'shared_patterns'])

                for i, comm in enumerate(communities, 1):
                    writer.writerow([
                        f"community_{i}",
                        len(comm['members']),
                        len(comm['shared_patterns']),
                        '|'.join(comm['emails']),
                        '|'.join(sorted(comm['shared_patterns']))
                    ])

            logger.info(f"Found {len(communities)} consciousness communities")

        return {
            'communities': communities,
            'total_communities': len(communities),
            'largest_community': max(communities, key=lambda c: len(c['members'])) if communities else None
        }