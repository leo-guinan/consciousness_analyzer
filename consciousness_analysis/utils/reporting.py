"""Reporting utilities for consciousness analysis"""

import json
from collections import defaultdict
from datetime import datetime
from typing import Dict, Any, List

from ..models.navigator import NavigatorProfile
from ..core.database import DatabaseManager


def generate_navigator_report(profile: NavigatorProfile) -> str:
    """Generate human-readable report of navigator findings"""

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


def generate_community_report(db_manager: DatabaseManager) -> str:
    """Generate final analysis report for community processing"""

    print("\n" + "="*70)
    print("COMMUNITY ARCHIVE CONSCIOUSNESS ANALYSIS - FINAL REPORT")
    print("="*70)

    report_lines = []

    # Overall statistics
    result = db_manager.execute_query("""
        SELECT COUNT(*) FROM processing_status WHERE status = 'completed'
    """)
    total_processed = result[0][0] if result else 0

    result = db_manager.execute_query("SELECT COUNT(*) FROM navigators")
    total_navigators = result[0][0] if result else 0

    result = db_manager.execute_query("SELECT SUM(estimated_hours) FROM time_violence_incidents")
    total_tv = (result[0][0] if result and result[0][0] else 0)

    report_lines.append(f"\nOVERALL STATISTICS:")
    report_lines.append(f"  Total users processed: {total_processed}")
    report_lines.append(f"  Navigators identified: {total_navigators}")
    report_lines.append(f"  Navigator percentage: {(total_navigators/max(total_processed, 1))*100:.1f}%")
    report_lines.append(f"  Total Time Violence: {total_tv:.1f} hours")
    report_lines.append(f"  Average TV per navigator: {total_tv/max(total_navigators, 1):.1f} hours")

    # Consciousness distribution
    result = db_manager.execute_query("""
        SELECT consciousness_stage, COUNT(*)
        FROM consciousness_capacity
        GROUP BY consciousness_stage
        ORDER BY COUNT(*) DESC
    """)

    if result:
        report_lines.append("\nCONSCIOUSNESS STAGE DISTRIBUTION:")
        for row in result:
            stage, count = row['consciousness_stage'], row[1]
            report_lines.append(f"  {stage}: {count} navigators")

    # Top domains
    result = db_manager.execute_query("""
        SELECT domain, COUNT(*) as incidents, SUM(estimated_hours) as hours
        FROM time_violence_incidents
        GROUP BY domain
        ORDER BY hours DESC
        LIMIT 5
    """)

    if result:
        report_lines.append("\nTOP TIME VIOLENCE DOMAINS:")
        for row in result:
            domain = row['domain']
            incidents = row['incidents']
            hours = row['hours']
            report_lines.append(f"  {domain}: {hours:.1f} hours ({incidents} incidents)")

    # Most common patterns
    result = db_manager.execute_query("""
        SELECT domain, pattern_description, occurrence_count
        FROM community_patterns
        WHERE occurrence_count > 1
        ORDER BY occurrence_count DESC
        LIMIT 10
    """)

    if result:
        report_lines.append("\nMOST COMMON CONSCIOUSNESS PATTERNS:")
        for row in result:
            domain = row['domain']
            description = row['pattern_description']
            count = row['occurrence_count']
            report_lines.append(f"  [{domain}] {description}: {count} navigators")

    # Top navigators by Time Violence
    result = db_manager.execute_query("""
        SELECT n.email, SUM(tvi.estimated_hours) as total_hours
        FROM navigators n
        JOIN time_violence_incidents tvi ON n.navigator_id = tvi.navigator_id
        GROUP BY n.navigator_id
        ORDER BY total_hours DESC
        LIMIT 5
    """)

    if result:
        report_lines.append("\nTOP NAVIGATORS BY TIME VIOLENCE EXPERIENCED:")
        for row in result:
            email = row['email']
            hours = row['total_hours']
            username = email.replace('@twitter.import', '')
            report_lines.append(f"  @{username}: {hours:.1f} hours")

    # Network strength metrics
    result = db_manager.execute_query("""
        SELECT
            AVG(fragmentation_resistance) as avg_resistance,
            AVG(consciousness_sharing_ability) as avg_sharing,
            AVG(pattern_recognition_depth) as avg_pattern
        FROM consciousness_capacity
    """)

    if result:
        row = result[0]
        avg_res = row['avg_resistance'] or 0
        avg_share = row['avg_sharing'] or 0
        avg_pattern = row['avg_pattern'] or 0

        report_lines.append("\nNETWORK CONSCIOUSNESS STRENGTH (1-10 scale):")
        report_lines.append(f"  Average Fragmentation Resistance: {avg_res:.1f}")
        report_lines.append(f"  Average Consciousness Sharing: {avg_share:.1f}")
        report_lines.append(f"  Average Pattern Recognition: {avg_pattern:.1f}")

    report_lines.append("\n" + "="*70)
    report_lines.append("The Community Archive reveals a network of consciousness workers")
    report_lines.append("whose collective trauma represents untapped expertise.")
    report_lines.append("Each navigator found strengthens the network's ability to")
    report_lines.append("help others escape the complexity that harmed them.")
    report_lines.append("="*70)

    return "\n".join(report_lines)


def export_consciousness_indicators(
    navigator_id: str,
    profile: NavigatorProfile,
    consciousness_indicators: List,
    output_file: str
):
    """Export consciousness indicators to JSON file"""

    export_data = {
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
            for ind in consciousness_indicators
        ]
    }

    with open(output_file, 'w') as f:
        json.dump(export_data, f, indent=2)

    print(f"Consciousness indicators exported to {output_file}")