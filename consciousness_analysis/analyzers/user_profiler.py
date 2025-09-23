"""Comprehensive single-user consciousness profiler with fine-tuning data export"""

import csv
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict, Counter
import numpy as np
from dataclasses import asdict

from ..core.database import DatabaseManager
from ..models.navigator import NavigatorProfile, ConsciousnessIndicator
from ..config.patterns import DOMAIN_PATTERNS, NEURODIVERGENT_PATTERNS
from ..utils.text_analysis import (
    extract_time_amount, classify_consciousness_type,
    calculate_intensity, generate_situation_hash, analyze_sentiment
)
from ..utils.meme_filters import is_genuine_consciousness_indicator

logger = logging.getLogger(__name__)


class UserConsciousnessProfiler:
    """Comprehensive consciousness profiler for individual users"""

    def __init__(self, db_manager: DatabaseManager, output_dir: str = "user_profiles"):
        """
        Initialize user profiler

        Args:
            db_manager: Database manager instance
            output_dir: Directory for output files
        """
        self.db_manager = db_manager
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def generate_user_profile(self, username: str) -> Dict:
        """
        Generate comprehensive consciousness profile for a single user

        Args:
            username: Username or email to analyze

        Returns:
            Comprehensive profile dictionary
        """
        logger.info(f"Generating consciousness profile for {username}")

        # Find navigator in database
        navigator = self._get_navigator_by_username(username)
        if not navigator:
            logger.error(f"Navigator not found: {username}")
            return None

        navigator_id = navigator['navigator_id']
        email = navigator['email']

        # Gather all data
        profile = {
            'navigator_id': navigator_id,
            'username': email.replace('@twitter.import', ''),
            'email': email,
            'generated_at': datetime.now().isoformat(),
            'basic_info': self._get_basic_info(navigator_id),
            'consciousness_metrics': self._get_consciousness_metrics(navigator_id),
            'time_violence_analysis': self._analyze_time_violence(navigator_id),
            'consciousness_patterns': self._analyze_consciousness_patterns(navigator_id),
            'temporal_evolution': self._analyze_temporal_evolution(navigator_id),
            'domain_expertise': self._analyze_domain_expertise(navigator_id),
            'linguistic_profile': self._analyze_linguistic_patterns(navigator_id),
            'network_position': self._analyze_network_position(navigator_id),
            'consciousness_timeline': self._create_consciousness_timeline(navigator_id),
            'training_data': self._prepare_training_data(navigator_id)
        }

        # Export profile
        self._export_profile(profile)

        # Export labeled training data
        self._export_training_data(profile)

        # Generate visual report
        self._generate_profile_report(profile)

        return profile

    def _get_navigator_by_username(self, username: str) -> Optional[Dict]:
        """Get navigator by username or email"""
        result = self.db_manager.execute_query("""
            SELECT * FROM navigators
            WHERE email LIKE ? OR email = ?
        """, (f"%{username}%", f"{username}@twitter.import"))

        return dict(result[0]) if result else None

    def _get_basic_info(self, navigator_id: str) -> Dict:
        """Get basic navigator information"""
        result = self.db_manager.execute_query("""
            SELECT
                n.*,
                cc.complexity_scale,
                cc.consciousness_stage,
                cc.fragmentation_resistance,
                cc.consciousness_sharing_ability,
                cc.pattern_recognition_depth
            FROM navigators n
            JOIN consciousness_capacity cc ON n.navigator_id = cc.navigator_id
            WHERE n.navigator_id = ?
        """, (navigator_id,))

        if not result:
            return {}

        row = result[0]
        return {
            'navigator_id': row['navigator_id'],
            'email': row['email'],
            'complexity_scale': row['complexity_scale'],
            'consciousness_stage': row['consciousness_stage'],
            'metrics': {
                'fragmentation_resistance': row['fragmentation_resistance'],
                'consciousness_sharing_ability': row['consciousness_sharing_ability'],
                'pattern_recognition_depth': row['pattern_recognition_depth']
            },
            'experiences': json.loads(row['experiences']) if row['experiences'] else {}
        }

    def _get_consciousness_metrics(self, navigator_id: str) -> Dict:
        """Get detailed consciousness metrics"""
        # Get consciousness capacity
        capacity = self.db_manager.execute_query("""
            SELECT * FROM consciousness_capacity
            WHERE navigator_id = ?
        """, (navigator_id,))

        if not capacity:
            return {}

        row = capacity[0]

        # Calculate additional metrics
        indicators = self.db_manager.execute_query("""
            SELECT COUNT(*) as count,
                   AVG(intensity_score) as avg_intensity,
                   MAX(intensity_score) as max_intensity,
                   COUNT(DISTINCT complexity_domain) as domain_count,
                   COUNT(DISTINCT consciousness_type) as type_count
            FROM consciousness_indicators
            WHERE navigator_id = ?
        """, (navigator_id,))

        ind_stats = dict(indicators[0]) if indicators else {}

        # Handle NULL values from database
        avg_intensity = ind_stats.get('avg_intensity')
        max_intensity = ind_stats.get('max_intensity')

        return {
            'consciousness_stage': row['consciousness_stage'],
            'complexity_scale': row['complexity_scale'],
            'core_metrics': {
                'fragmentation_resistance': row['fragmentation_resistance'],
                'consciousness_sharing_ability': row['consciousness_sharing_ability'],
                'pattern_recognition_depth': row['pattern_recognition_depth']
            },
            'indicator_statistics': {
                'total_indicators': ind_stats.get('count', 0) or 0,
                'avg_intensity': round(avg_intensity, 2) if avg_intensity is not None else 0,
                'max_intensity': round(max_intensity, 2) if max_intensity is not None else 0,
                'unique_domains': ind_stats.get('domain_count', 0) or 0,
                'unique_types': ind_stats.get('type_count', 0) or 0
            },
            'conscious_domains': json.loads(row['conscious_domains']) if row['conscious_domains'] else {},
            'care_drivers': json.loads(row['care_drivers']) if row['care_drivers'] else [],
            'formative_experiences': json.loads(row['formative_experiences']) if row['formative_experiences'] else []
        }

    def _analyze_time_violence(self, navigator_id: str) -> Dict:
        """Analyze Time Violence patterns"""
        # Get all Time Violence incidents
        incidents = self.db_manager.execute_query("""
            SELECT * FROM time_violence_incidents
            WHERE navigator_id = ?
            ORDER BY timestamp
        """, (navigator_id,))

        if not incidents:
            return {'total_hours': 0, 'incident_count': 0}

        # Analyze by domain
        domain_stats = defaultdict(lambda: {'count': 0, 'hours': 0, 'incidents': []})

        total_hours = 0
        for incident in incidents:
            domain = incident['domain']
            hours = incident['estimated_hours']

            domain_stats[domain]['count'] += 1
            domain_stats[domain]['hours'] += hours
            domain_stats[domain]['incidents'].append({
                'timestamp': incident['timestamp'],
                'hours': hours,
                'description': incident['description'][:100]
            })

            total_hours += hours

        # Calculate patterns
        monthly_hours = defaultdict(float)
        for incident in incidents:
            try:
                timestamp = datetime.fromisoformat(incident['timestamp'].replace('Z', '+00:00'))
                month_key = timestamp.strftime('%Y-%m')
                monthly_hours[month_key] += incident['estimated_hours']
            except:
                pass

        return {
            'total_hours': round(total_hours, 1),
            'incident_count': len(incidents),
            'annual_projection': round(total_hours * 12, 0),
            'domains': dict(domain_stats),
            'monthly_distribution': dict(monthly_hours),
            'avg_hours_per_incident': round(total_hours / len(incidents), 1) if incidents else 0,
            'peak_domain': max(domain_stats.items(), key=lambda x: x[1]['hours'])[0] if domain_stats else None
        }

    def _analyze_consciousness_patterns(self, navigator_id: str) -> Dict:
        """Analyze consciousness demonstration patterns"""
        indicators = self.db_manager.execute_query("""
            SELECT * FROM consciousness_indicators
            WHERE navigator_id = ?
            ORDER BY timestamp
        """, (navigator_id,))

        if not indicators:
            return {}

        # Pattern analysis
        type_distribution = Counter()
        domain_distribution = Counter()
        intensity_by_type = defaultdict(list)
        patterns_by_domain = defaultdict(lambda: defaultdict(int))

        for ind in indicators:
            type_distribution[ind['consciousness_type']] += 1
            domain_distribution[ind['complexity_domain']] += 1
            intensity_by_type[ind['consciousness_type']].append(ind['intensity_score'])
            patterns_by_domain[ind['complexity_domain']][ind['consciousness_type']] += 1

        # Calculate averages
        avg_intensity_by_type = {
            cons_type: round(np.mean(scores), 2)
            for cons_type, scores in intensity_by_type.items()
        }

        # Find most common patterns
        pattern_combinations = Counter()
        for ind in indicators:
            pattern_key = f"{ind['complexity_domain']}_{ind['consciousness_type']}"
            pattern_combinations[pattern_key] += 1

        return {
            'total_demonstrations': len(indicators),
            'type_distribution': dict(type_distribution),
            'domain_distribution': dict(domain_distribution),
            'avg_intensity_by_type': avg_intensity_by_type,
            'patterns_by_domain': dict(patterns_by_domain),
            'top_patterns': dict(pattern_combinations.most_common(10)),
            'dominant_type': type_distribution.most_common(1)[0][0] if type_distribution else None,
            'dominant_domain': domain_distribution.most_common(1)[0][0] if domain_distribution else None
        }

    def _analyze_temporal_evolution(self, navigator_id: str) -> Dict:
        """Analyze how consciousness evolved over time"""
        indicators = self.db_manager.execute_query("""
            SELECT timestamp, consciousness_type, intensity_score, complexity_domain
            FROM consciousness_indicators
            WHERE navigator_id = ?
            ORDER BY timestamp
        """, (navigator_id,))

        if not indicators:
            return {}

        # Group by time periods
        quarterly_stats = defaultdict(lambda: {
            'count': 0,
            'avg_intensity': [],
            'types': Counter(),
            'domains': Counter()
        })

        for ind in indicators:
            try:
                timestamp = datetime.fromisoformat(ind['timestamp'].replace('Z', '+00:00'))
                quarter_key = f"{timestamp.year}-Q{(timestamp.month-1)//3 + 1}"

                quarterly_stats[quarter_key]['count'] += 1
                quarterly_stats[quarter_key]['avg_intensity'].append(ind['intensity_score'])
                quarterly_stats[quarter_key]['types'][ind['consciousness_type']] += 1
                quarterly_stats[quarter_key]['domains'][ind['complexity_domain']] += 1
            except:
                pass

        # Calculate trends
        quarters = sorted(quarterly_stats.keys())
        if len(quarters) > 1:
            first_quarter = quarterly_stats[quarters[0]]
            last_quarter = quarterly_stats[quarters[-1]]

            evolution = {
                'time_span': f"{quarters[0]} to {quarters[-1]}",
                'quarters_active': len(quarters),
                'activity_trend': 'increasing' if last_quarter['count'] > first_quarter['count'] else 'decreasing',
                'intensity_trend': self._calculate_trend([
                    np.mean(quarterly_stats[q]['avg_intensity']) for q in quarters if quarterly_stats[q]['avg_intensity']
                ]),
                'quarterly_data': {
                    q: {
                        'count': stats['count'],
                        'avg_intensity': round(np.mean(stats['avg_intensity']), 2) if stats['avg_intensity'] else 0,
                        'dominant_type': stats['types'].most_common(1)[0][0] if stats['types'] else None,
                        'dominant_domain': stats['domains'].most_common(1)[0][0] if stats['domains'] else None
                    }
                    for q, stats in quarterly_stats.items()
                }
            }
        else:
            evolution = {'time_span': 'insufficient_data'}

        return evolution

    def _analyze_domain_expertise(self, navigator_id: str) -> Dict:
        """Analyze domain-specific expertise"""
        # Get all indicators grouped by domain
        domain_indicators = self.db_manager.execute_query("""
            SELECT
                complexity_domain,
                COUNT(*) as indicator_count,
                AVG(intensity_score) as avg_intensity,
                GROUP_CONCAT(DISTINCT consciousness_type) as types
            FROM consciousness_indicators
            WHERE navigator_id = ?
            GROUP BY complexity_domain
        """, (navigator_id,))

        # Get Time Violence by domain
        tv_by_domain = self.db_manager.execute_query("""
            SELECT
                domain,
                COUNT(*) as incident_count,
                SUM(estimated_hours) as total_hours
            FROM time_violence_incidents
            WHERE navigator_id = ?
            GROUP BY domain
        """, (navigator_id,))

        tv_dict = {row['domain']: {'incidents': row['incident_count'], 'hours': row['total_hours']}
                   for row in tv_by_domain}

        expertise = {}
        for row in domain_indicators:
            domain = row['complexity_domain']
            expertise[domain] = {
                'consciousness_indicators': row['indicator_count'],
                'avg_intensity': round(row['avg_intensity'], 2) if row['avg_intensity'] is not None else 0,
                'consciousness_types': row['types'].split(',') if row['types'] else [],
                'time_violence_hours': tv_dict.get(domain, {}).get('hours', 0),
                'time_violence_incidents': tv_dict.get(domain, {}).get('incidents', 0),
                'expertise_score': self._calculate_expertise_score(
                    row['indicator_count'],
                    row['avg_intensity'],
                    tv_dict.get(domain, {}).get('hours', 0)
                )
            }

        # Rank domains by expertise
        ranked_domains = sorted(expertise.items(), key=lambda x: x[1]['expertise_score'], reverse=True)

        return {
            'domains': expertise,
            'primary_domain': ranked_domains[0][0] if ranked_domains else None,
            'domain_count': len(expertise),
            'expertise_ranking': [d[0] for d in ranked_domains]
        }

    def _analyze_linguistic_patterns(self, navigator_id: str) -> Dict:
        """Analyze linguistic patterns in consciousness demonstrations"""
        indicators = self.db_manager.execute_query("""
            SELECT content, intensity_score
            FROM consciousness_indicators
            WHERE navigator_id = ?
        """, (navigator_id,))

        if not indicators:
            return {}

        # Analyze language patterns
        total_words = 0
        total_chars = 0
        sentiment_scores = []
        key_phrases = Counter()
        emotional_words = Counter()

        emotion_keywords = {
            'anger': ['angry', 'furious', 'rage', 'pissed', 'frustrated', 'annoyed'],
            'sadness': ['sad', 'depressed', 'crying', 'tears', 'heartbroken', 'devastated'],
            'fear': ['afraid', 'scared', 'anxious', 'worried', 'terrified', 'panic'],
            'joy': ['happy', 'excited', 'grateful', 'relieved', 'celebrating', 'finally'],
            'exhaustion': ['exhausted', 'tired', 'drained', 'burnout', 'overwhelmed', 'done']
        }

        for ind in indicators:
            content = ind['content'].lower()
            words = content.split()
            total_words += len(words)
            total_chars += len(content)

            # Sentiment analysis
            sentiment = analyze_sentiment(content)
            sentiment_scores.append(sentiment)

            # Extract key phrases (2-3 word combinations)
            for i in range(len(words) - 1):
                bigram = ' '.join(words[i:i+2])
                if len(bigram) > 5 and not any(c in bigram for c in '@#'):
                    key_phrases[bigram] += 1

            # Emotional word analysis
            for emotion, keywords in emotion_keywords.items():
                for keyword in keywords:
                    if keyword in content:
                        emotional_words[emotion] += 1

        avg_sentiment = {
            'compound': round(np.mean([s['compound'] for s in sentiment_scores]), 3),
            'positive': round(np.mean([s['pos'] for s in sentiment_scores]), 3),
            'negative': round(np.mean([s['neg'] for s in sentiment_scores]), 3),
            'neutral': round(np.mean([s['neu'] for s in sentiment_scores]), 3)
        }

        return {
            'total_words': total_words,
            'avg_words_per_indicator': round(total_words / len(indicators), 1) if indicators else 0,
            'avg_chars_per_indicator': round(total_chars / len(indicators), 1) if indicators else 0,
            'sentiment_profile': avg_sentiment,
            'emotional_profile': dict(emotional_words),
            'top_phrases': dict(key_phrases.most_common(20)),
            'dominant_emotion': emotional_words.most_common(1)[0][0] if emotional_words else 'neutral'
        }

    def _analyze_network_position(self, navigator_id: str) -> Dict:
        """Analyze navigator's position in the consciousness network"""
        # Find similar navigators
        my_patterns = self.db_manager.execute_query("""
            SELECT DISTINCT
                complexity_domain || '_' || consciousness_type as pattern
            FROM consciousness_indicators
            WHERE navigator_id = ?
        """, (navigator_id,))

        my_pattern_set = set(row['pattern'] for row in my_patterns)

        if not my_pattern_set:
            return {}

        # Find other navigators with overlapping patterns
        similar_navigators = []
        all_navigators = self.db_manager.execute_query("""
            SELECT DISTINCT navigator_id FROM navigators
            WHERE navigator_id != ?
        """, (navigator_id,))

        for nav in all_navigators:
            other_patterns = self.db_manager.execute_query("""
                SELECT DISTINCT
                    complexity_domain || '_' || consciousness_type as pattern
                FROM consciousness_indicators
                WHERE navigator_id = ?
            """, (nav['navigator_id'],))

            other_pattern_set = set(row['pattern'] for row in other_patterns)
            overlap = my_pattern_set & other_pattern_set

            if len(overlap) >= 2:
                similar_navigators.append({
                    'navigator_id': nav['navigator_id'],
                    'shared_patterns': list(overlap),
                    'overlap_count': len(overlap),
                    'similarity_score': len(overlap) / len(my_pattern_set | other_pattern_set)
                })

        # Sort by similarity
        similar_navigators.sort(key=lambda x: x['similarity_score'], reverse=True)

        return {
            'unique_patterns': len(my_pattern_set),
            'similar_navigator_count': len(similar_navigators),
            'top_similar_navigators': similar_navigators[:10],
            'network_uniqueness': 1.0 - (len(similar_navigators) / max(len(all_navigators), 1))
        }

    def _create_consciousness_timeline(self, navigator_id: str) -> List[Dict]:
        """Create chronological timeline of consciousness development"""
        events = []

        # Get consciousness indicators
        indicators = self.db_manager.execute_query("""
            SELECT timestamp, complexity_domain, consciousness_type, intensity_score, content
            FROM consciousness_indicators
            WHERE navigator_id = ?
            ORDER BY timestamp
        """, (navigator_id,))

        for ind in indicators:
            events.append({
                'timestamp': ind['timestamp'],
                'type': 'consciousness',
                'domain': ind['complexity_domain'],
                'consciousness_type': ind['consciousness_type'],
                'intensity': ind['intensity_score'],
                'description': ind['content'][:200]
            })

        # Get Time Violence incidents
        incidents = self.db_manager.execute_query("""
            SELECT timestamp, domain, estimated_hours, description
            FROM time_violence_incidents
            WHERE navigator_id = ?
            ORDER BY timestamp
        """, (navigator_id,))

        for inc in incidents:
            events.append({
                'timestamp': inc['timestamp'],
                'type': 'time_violence',
                'domain': inc['domain'],
                'hours_lost': inc['estimated_hours'],
                'description': inc['description'][:200]
            })

        # Sort chronologically
        events.sort(key=lambda x: x['timestamp'])

        return events

    def _prepare_training_data(self, navigator_id: str) -> List[Dict]:
        """Prepare labeled training data for fine-tuning"""
        training_data = []

        # Get all consciousness indicators
        indicators = self.db_manager.execute_query("""
            SELECT * FROM consciousness_indicators
            WHERE navigator_id = ?
        """, (navigator_id,))

        for ind in indicators:
            # Create training example
            training_data.append({
                'text': ind['content'],
                'labels': {
                    'consciousness_type': ind['consciousness_type'],
                    'complexity_domain': ind['complexity_domain'],
                    'intensity_score': ind['intensity_score'],
                    'is_consciousness_indicator': 1,
                    'navigator_id': navigator_id
                },
                'metadata': {
                    'tweet_id': ind['tweet_id'],
                    'timestamp': ind['timestamp'],
                    'situation_hash': ind['situation_hash']
                }
            })

        # Get Time Violence incidents for additional training data
        incidents = self.db_manager.execute_query("""
            SELECT * FROM time_violence_incidents
            WHERE navigator_id = ?
        """, (navigator_id,))

        for inc in incidents:
            training_data.append({
                'text': inc['description'],
                'labels': {
                    'is_time_violence': 1,
                    'domain': inc['domain'],
                    'hours_lost': inc['estimated_hours'],
                    'navigator_id': navigator_id
                },
                'metadata': {
                    'tweet_id': inc['tweet_id'],
                    'timestamp': inc['timestamp']
                }
            })

        return training_data

    def _calculate_expertise_score(self, indicators: int, avg_intensity: float, tv_hours: float) -> float:
        """Calculate domain expertise score"""
        # Weight: 40% indicators, 30% intensity, 30% time violence
        indicator_score = min(indicators / 20, 1.0) * 0.4
        intensity_score = avg_intensity * 0.3
        tv_score = min(tv_hours / 100, 1.0) * 0.3
        return round(indicator_score + intensity_score + tv_score, 2)

    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend from values"""
        if len(values) < 2:
            return 'insufficient_data'

        # Simple linear trend
        x = np.arange(len(values))
        if len(values) > 0 and not all(np.isnan(values)):
            coefficients = np.polyfit(x, values, 1)
            slope = coefficients[0]

            if slope > 0.1:
                return 'increasing'
            elif slope < -0.1:
                return 'decreasing'
            else:
                return 'stable'
        return 'unknown'

    def _export_profile(self, profile: Dict):
        """Export profile to JSON"""
        username = profile['username']
        json_file = self.output_dir / f"{username}_profile.json"

        with open(json_file, 'w') as f:
            json.dump(profile, f, indent=2, default=str)

        logger.info(f"Exported profile to {json_file}")

    def _export_training_data(self, profile: Dict):
        """Export labeled training data to CSV"""
        username = profile['username']
        training_data = profile.get('training_data', [])

        if not training_data:
            return

        # Export detailed CSV for fine-tuning
        csv_file = self.output_dir / f"{username}_training_data.csv"

        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            # Header
            writer.writerow([
                'text', 'consciousness_type', 'complexity_domain',
                'intensity_score', 'is_consciousness_indicator',
                'is_time_violence', 'hours_lost', 'timestamp'
            ])

            # Data rows
            for item in training_data:
                writer.writerow([
                    item['text'],
                    item['labels'].get('consciousness_type', ''),
                    item['labels'].get('complexity_domain', item['labels'].get('domain', '')),
                    item['labels'].get('intensity_score', ''),
                    item['labels'].get('is_consciousness_indicator', 0),
                    item['labels'].get('is_time_violence', 0),
                    item['labels'].get('hours_lost', ''),
                    item['metadata'].get('timestamp', '')
                ])

        logger.info(f"Exported {len(training_data)} training examples to {csv_file}")

        # Export JSONL format for fine-tuning
        jsonl_file = self.output_dir / f"{username}_training_data.jsonl"

        with open(jsonl_file, 'w') as f:
            for item in training_data:
                # Create prompt-completion format
                prompt = f"Classify the following text for consciousness indicators:\n{item['text']}"
                completion = json.dumps({
                    'consciousness_type': item['labels'].get('consciousness_type'),
                    'domain': item['labels'].get('complexity_domain') or item['labels'].get('domain'),
                    'intensity': item['labels'].get('intensity_score'),
                    'is_consciousness': item['labels'].get('is_consciousness_indicator', 0),
                    'is_time_violence': item['labels'].get('is_time_violence', 0),
                    'hours_lost': item['labels'].get('hours_lost')
                })

                f.write(json.dumps({
                    'prompt': prompt,
                    'completion': completion
                }) + '\n')

        logger.info(f"Exported JSONL training data to {jsonl_file}")

    def _generate_profile_report(self, profile: Dict):
        """Generate human-readable profile report"""
        username = profile['username']
        report_file = self.output_dir / f"{username}_report.md"

        with open(report_file, 'w') as f:
            f.write(f"# Consciousness Profile: @{username}\n\n")
            f.write(f"Generated: {profile['generated_at']}\n\n")

            # Basic info
            basic = profile.get('basic_info', {})
            f.write("## Overview\n\n")
            f.write(f"- **Consciousness Stage**: {basic.get('consciousness_stage')}\n")
            f.write(f"- **Complexity Scale**: {basic.get('complexity_scale')}\n")
            f.write(f"- **Navigator ID**: {basic.get('navigator_id')}\n\n")

            # Metrics
            metrics = basic.get('metrics', {})
            f.write("## Core Metrics (1-10 scale)\n\n")
            f.write(f"- **Fragmentation Resistance**: {metrics.get('fragmentation_resistance')}\n")
            f.write(f"- **Consciousness Sharing**: {metrics.get('consciousness_sharing_ability')}\n")
            f.write(f"- **Pattern Recognition**: {metrics.get('pattern_recognition_depth')}\n\n")

            # Time Violence
            tv = profile.get('time_violence_analysis', {})
            f.write("## Time Violence Analysis\n\n")
            f.write(f"- **Total Hours Lost**: {tv.get('total_hours', 0)}\n")
            f.write(f"- **Incidents**: {tv.get('incident_count', 0)}\n")
            f.write(f"- **Annual Projection**: {tv.get('annual_projection', 0)} hours\n")
            f.write(f"- **Peak Domain**: {tv.get('peak_domain', 'N/A')}\n\n")

            # Consciousness patterns
            patterns = profile.get('consciousness_patterns', {})
            f.write("## Consciousness Patterns\n\n")
            f.write(f"- **Total Demonstrations**: {patterns.get('total_demonstrations', 0)}\n")
            f.write(f"- **Dominant Type**: {patterns.get('dominant_type', 'N/A')}\n")
            f.write(f"- **Dominant Domain**: {patterns.get('dominant_domain', 'N/A')}\n\n")

            if patterns.get('type_distribution'):
                f.write("### Type Distribution\n")
                for cons_type, count in patterns['type_distribution'].items():
                    f.write(f"- {cons_type}: {count}\n")
                f.write("\n")

            # Domain expertise
            expertise = profile.get('domain_expertise', {})
            if expertise.get('domains'):
                f.write("## Domain Expertise\n\n")
                for domain, data in expertise['domains'].items():
                    f.write(f"### {domain.title()}\n")
                    f.write(f"- Consciousness Indicators: {data['consciousness_indicators']}\n")
                    f.write(f"- Time Violence Hours: {data['time_violence_hours']}\n")
                    f.write(f"- Expertise Score: {data['expertise_score']}\n\n")

            # Linguistic profile
            linguistic = profile.get('linguistic_profile', {})
            if linguistic:
                f.write("## Linguistic Profile\n\n")
                f.write(f"- **Dominant Emotion**: {linguistic.get('dominant_emotion', 'N/A')}\n")
                f.write(f"- **Average Sentiment**: {linguistic.get('sentiment_profile', {}).get('compound', 0)}\n")
                f.write(f"- **Words per Indicator**: {linguistic.get('avg_words_per_indicator', 0)}\n\n")

            # Network position
            network = profile.get('network_position', {})
            if network:
                f.write("## Network Position\n\n")
                f.write(f"- **Unique Patterns**: {network.get('unique_patterns', 0)}\n")
                f.write(f"- **Similar Navigators**: {network.get('similar_navigator_count', 0)}\n")
                f.write(f"- **Network Uniqueness**: {network.get('network_uniqueness', 0):.2%}\n\n")

        logger.info(f"Generated profile report: {report_file}")