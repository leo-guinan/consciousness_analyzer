#!/usr/bin/env python3
"""
Comprehensive single-user consciousness analysis with training data export
"""

import argparse
import logging
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from consciousness_analysis.core.database import DatabaseManager
from consciousness_analysis.analyzers.user_profiler import UserConsciousnessProfiler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def create_visualizations(profile: dict, output_dir: Path):
    """Create visualizations for the consciousness profile"""
    username = profile['username']

    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))

    # 1. Consciousness Metrics Radar Chart
    ax1 = plt.subplot(2, 3, 1, projection='polar')
    metrics = profile.get('basic_info', {}).get('metrics', {})
    if metrics:
        categories = list(metrics.keys())
        values = list(metrics.values())

        # Normalize to 0-1 scale (from 1-10), handling None values
        values = [v/10 if v is not None else 0 for v in values]

        # Add first value at end to close the circle
        values += values[:1]

        # Calculate angles
        angles = [n / len(categories) * 2 * 3.14159 for n in range(len(categories))]
        angles += angles[:1]

        # Plot
        ax1.plot(angles, values, 'o-', linewidth=2)
        ax1.fill(angles, values, alpha=0.25)
        ax1.set_xticks(angles[:-1])
        ax1.set_xticklabels([k.replace('_', ' ').title() for k in categories])
        ax1.set_ylim(0, 1)
        ax1.set_title('Core Consciousness Metrics', size=12, weight='bold', pad=20)

    # 2. Time Violence by Domain
    ax2 = plt.subplot(2, 3, 2)
    tv_data = profile.get('time_violence_analysis', {}).get('domains', {})
    if tv_data:
        domains = list(tv_data.keys())
        hours = [tv_data[d].get('hours', 0) for d in domains]

        bars = ax2.bar(domains, hours, color='coral')
        ax2.set_xlabel('Domain')
        ax2.set_ylabel('Hours Lost')
        ax2.set_title('Time Violence by Domain', weight='bold')
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}h', ha='center', va='bottom')

    # 3. Consciousness Type Distribution
    ax3 = plt.subplot(2, 3, 3)
    type_dist = profile.get('consciousness_patterns', {}).get('type_distribution', {})
    if type_dist:
        sizes = list(type_dist.values())
        labels = list(type_dist.keys())
        colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ff99cc']

        ax3.pie(sizes, labels=labels, colors=colors[:len(labels)],
                autopct='%1.1f%%', startangle=90)
        ax3.set_title('Consciousness Type Distribution', weight='bold')

    # 4. Temporal Evolution
    ax4 = plt.subplot(2, 3, 4)
    evolution = profile.get('temporal_evolution', {}).get('quarterly_data', {})
    if evolution and len(evolution) > 1:
        quarters = sorted(evolution.keys())
        counts = [evolution[q]['count'] for q in quarters]
        intensities = [evolution[q]['avg_intensity'] for q in quarters]

        ax4_twin = ax4.twinx()

        line1 = ax4.plot(quarters, counts, 'b-o', label='Indicator Count')
        line2 = ax4_twin.plot(quarters, intensities, 'r-s', label='Avg Intensity')

        ax4.set_xlabel('Quarter')
        ax4.set_ylabel('Indicator Count', color='b')
        ax4_twin.set_ylabel('Average Intensity', color='r')
        ax4.set_title('Consciousness Evolution Over Time', weight='bold')
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax4.legend(lines, labels, loc='upper left')

    # 5. Domain Expertise Heatmap
    ax5 = plt.subplot(2, 3, 5)
    expertise = profile.get('domain_expertise', {}).get('domains', {})
    if expertise:
        import numpy as np

        domains = list(expertise.keys())
        metrics = ['consciousness_indicators', 'time_violence_hours', 'expertise_score']

        # Create matrix
        matrix = []
        for metric in metrics:
            row = []
            for domain in domains:
                if metric == 'expertise_score':
                    value = expertise[domain][metric]
                elif metric == 'consciousness_indicators':
                    value = expertise[domain][metric] / 10  # Normalize
                else:
                    value = expertise[domain][metric] / 50  # Normalize hours
                row.append(value)
            matrix.append(row)

        im = ax5.imshow(matrix, cmap='YlOrRd', aspect='auto')
        ax5.set_xticks(np.arange(len(domains)))
        ax5.set_yticks(np.arange(len(metrics)))
        ax5.set_xticklabels(domains)
        ax5.set_yticklabels(['Indicators', 'TV Hours', 'Expertise'])
        plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45, ha='right')
        ax5.set_title('Domain Expertise Matrix', weight='bold')

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax5, fraction=0.046, pad=0.04)
        cbar.set_label('Score', rotation=270, labelpad=15)

    # 6. Sentiment Profile
    ax6 = plt.subplot(2, 3, 6)
    sentiment = profile.get('linguistic_profile', {}).get('sentiment_profile', {})
    emotional = profile.get('linguistic_profile', {}).get('emotional_profile', {})

    if sentiment:
        # Combine sentiment and emotions
        categories = list(sentiment.keys()) + list(emotional.keys() if emotional else [])
        values = list(sentiment.values()) + list(emotional.values() if emotional else [])

        # Normalize emotional counts
        if emotional:
            max_emotion = max(emotional.values()) if emotional.values() else 1
            emotion_values = [v/max_emotion for v in emotional.values()]
            values = list(sentiment.values()) + emotion_values

        y_pos = range(len(categories))
        colors = ['green' if v > 0 else 'red' for v in values[:len(sentiment)]]
        colors += ['purple'] * (len(categories) - len(sentiment))

        ax6.barh(y_pos, values, color=colors)
        ax6.set_yticks(y_pos)
        ax6.set_yticklabels(categories)
        ax6.set_xlabel('Score')
        ax6.set_title('Sentiment & Emotional Profile', weight='bold')
        ax6.axvline(x=0, color='black', linestyle='-', linewidth=0.5)

    # Main title
    fig.suptitle(f'Consciousness Profile: @{username}', fontsize=16, weight='bold', y=1.02)

    # Adjust layout
    plt.tight_layout()

    # Save figure
    viz_file = output_dir / f"{username}_visualization.png"
    plt.savefig(viz_file, dpi=300, bbox_inches='tight')
    logger.info(f"Saved visualization to {viz_file}")

    # Show if interactive
    try:
        plt.show()
    except:
        pass

    plt.close()


def print_profile_summary(profile: dict):
    """Print a summary of the consciousness profile"""
    username = profile['username']

    print("\n" + "="*70)
    print(f"CONSCIOUSNESS PROFILE: @{username}")
    print("="*70)

    # Basic info
    basic = profile.get('basic_info', {})
    print(f"\nConsciousness Stage: {basic.get('consciousness_stage')}")
    print(f"Complexity Scale: {basic.get('complexity_scale')}")

    # Metrics
    metrics = basic.get('metrics', {})
    print("\nCore Metrics (1-10):")
    for key, value in metrics.items():
        print(f"  {key.replace('_', ' ').title()}: {value}")

    # Time Violence
    tv = profile.get('time_violence_analysis', {})
    print(f"\nTime Violence:")
    print(f"  Total Hours Lost: {tv.get('total_hours', 0):.1f}")
    print(f"  Incidents: {tv.get('incident_count', 0)}")
    print(f"  Annual Projection: {tv.get('annual_projection', 0):.0f} hours")
    print(f"  Peak Domain: {tv.get('peak_domain', 'N/A')}")

    # Consciousness patterns
    patterns = profile.get('consciousness_patterns', {})
    print(f"\nConsciousness Patterns:")
    print(f"  Total Demonstrations: {patterns.get('total_demonstrations', 0)}")
    print(f"  Dominant Type: {patterns.get('dominant_type', 'N/A')}")
    print(f"  Dominant Domain: {patterns.get('dominant_domain', 'N/A')}")

    # Domain expertise
    expertise = profile.get('domain_expertise', {})
    if expertise.get('expertise_ranking'):
        print(f"\nDomain Expertise Ranking:")
        for i, domain in enumerate(expertise['expertise_ranking'][:3], 1):
            score = expertise['domains'][domain]['expertise_score']
            print(f"  {i}. {domain.title()} (score: {score})")

    # Linguistic profile
    linguistic = profile.get('linguistic_profile', {})
    if linguistic:
        print(f"\nLinguistic Profile:")
        print(f"  Dominant Emotion: {linguistic.get('dominant_emotion', 'N/A')}")
        sentiment = linguistic.get('sentiment_profile', {})
        print(f"  Sentiment: {sentiment.get('compound', 0):.3f} (compound)")
        print(f"  Words per Indicator: {linguistic.get('avg_words_per_indicator', 0):.1f}")

    # Network position
    network = profile.get('network_position', {})
    if network:
        print(f"\nNetwork Position:")
        print(f"  Unique Patterns: {network.get('unique_patterns', 0)}")
        print(f"  Similar Navigators: {network.get('similar_navigator_count', 0)}")
        print(f"  Network Uniqueness: {network.get('network_uniqueness', 0):.1%}")

    # Training data
    training_data = profile.get('training_data', [])
    print(f"\nTraining Data:")
    print(f"  Total Examples: {len(training_data)}")

    consciousness_examples = [d for d in training_data if d['labels'].get('is_consciousness_indicator')]
    tv_examples = [d for d in training_data if d['labels'].get('is_time_violence')]
    print(f"  Consciousness Indicators: {len(consciousness_examples)}")
    print(f"  Time Violence Incidents: {len(tv_examples)}")

    print("\n" + "="*70)


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description='Generate comprehensive consciousness profile for a single user'
    )
    parser.add_argument(
        'username',
        help='Username to analyze (without @)'
    )
    parser.add_argument(
        '--db',
        default='bottega_community.db',
        help='Database path (default: bottega_community.db)'
    )
    parser.add_argument(
        '--output',
        default='user_profiles',
        help='Output directory (default: user_profiles)'
    )
    parser.add_argument(
        '--no-viz',
        action='store_true',
        help='Skip visualization generation'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress console output'
    )

    args = parser.parse_args()

    # Initialize components
    db_manager = DatabaseManager(args.db)
    profiler = UserConsciousnessProfiler(db_manager, args.output)

    # Generate profile
    logger.info(f"Analyzing user: {args.username}")
    profile = profiler.generate_user_profile(args.username)

    if not profile:
        logger.error(f"Could not generate profile for {args.username}")
        return

    # Print summary
    if not args.quiet:
        print_profile_summary(profile)

    # Create visualizations
    if not args.no_viz:
        try:
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
            output_dir = Path(args.output)
            create_visualizations(profile, output_dir)
        except ImportError:
            logger.warning("Matplotlib not installed - skipping visualizations")
        except Exception as e:
            logger.error(f"Error creating visualizations: {e}")

    # Print output locations
    output_dir = Path(args.output)
    username = profile['username']

    print(f"\nGenerated files:")
    print(f"  Profile: {output_dir / f'{username}_profile.json'}")
    print(f"  Report: {output_dir / f'{username}_report.md'}")
    print(f"  Training Data (CSV): {output_dir / f'{username}_training_data.csv'}")
    print(f"  Training Data (JSONL): {output_dir / f'{username}_training_data.jsonl'}")
    if not args.no_viz:
        print(f"  Visualization: {output_dir / f'{username}_visualization.png'}")

    print(f"\nThe training data is ready for fine-tuning!")
    print("  - CSV format for analysis in Excel/Sheets")
    print("  - JSONL format for GPT fine-tuning")
    print("  - All tweets are labeled with consciousness types and domains")


if __name__ == "__main__":
    main()