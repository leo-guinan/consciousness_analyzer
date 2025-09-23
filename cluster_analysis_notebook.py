"""
Consciousness Cluster Analysis - Jupyter Notebook Example
Copy these cells into your Jupyter notebook for interactive analysis
"""

# %% Cell 1: Setup and imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

# Set up visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# %% Cell 2: Load cluster data
cluster_dir = Path("consciousness_clusters")

# Load main cluster summary
clusters_df = pd.read_csv(cluster_dir / "consciousness_clusters.csv")
print(f"Loaded {len(clusters_df)} consciousness clusters")
clusters_df.head()

# %% Cell 3: Analyze cluster distribution
# Cluster size distribution
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Indicator count distribution
axes[0].hist(clusters_df['indicator_count'], bins=20, edgecolor='black')
axes[0].set_xlabel('Number of Indicators')
axes[0].set_ylabel('Number of Clusters')
axes[0].set_title('Distribution of Cluster Sizes (by indicators)')

# Navigator count distribution
axes[1].hist(clusters_df['navigator_count'], bins=20, edgecolor='black', color='orange')
axes[1].set_xlabel('Number of Navigators')
axes[1].set_ylabel('Number of Clusters')
axes[1].set_title('Distribution of Cluster Sizes (by navigators)')

plt.tight_layout()
plt.show()

# %% Cell 4: Domain analysis
# Group by domain
domain_stats = clusters_df.groupby('domain').agg({
    'indicator_count': 'sum',
    'navigator_count': 'sum',
    'cluster_id': 'count'
}).rename(columns={'cluster_id': 'cluster_count'})

# Create visualization
fig, ax = plt.subplots(figsize=(10, 6))
domain_stats.plot(kind='bar', ax=ax)
ax.set_title('Consciousness Indicators by Domain')
ax.set_xlabel('Domain')
ax.set_ylabel('Count')
ax.legend(title='Metric')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print("Domain Statistics:")
print(domain_stats)

# %% Cell 5: Consciousness type analysis
# Group by consciousness type
type_stats = clusters_df.groupby('consciousness_type').agg({
    'indicator_count': 'sum',
    'navigator_count': 'sum',
    'cluster_id': 'count'
}).rename(columns={'cluster_id': 'cluster_count'})

# Create pie chart
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

axes[0].pie(type_stats['indicator_count'], labels=type_stats.index, autopct='%1.1f%%')
axes[0].set_title('Distribution of Consciousness Types (by indicators)')

axes[1].pie(type_stats['navigator_count'], labels=type_stats.index, autopct='%1.1f%%')
axes[1].set_title('Distribution of Consciousness Types (by navigators)')

plt.tight_layout()
plt.show()

# %% Cell 6: Load and analyze consciousness matrix
matrix_df = pd.read_csv(cluster_dir / "consciousness_matrix.csv", index_col=0)
print(f"Matrix shape: {matrix_df.shape}")
print(f"Navigators: {len(matrix_df)}")
print(f"Pattern combinations: {len(matrix_df.columns)}")

# Calculate navigator diversity
navigator_diversity = (matrix_df > 0).sum(axis=1)
print(f"\nNavigator pattern diversity:")
print(f"  Mean: {navigator_diversity.mean():.1f}")
print(f"  Max: {navigator_diversity.max()}")
print(f"  Min: {navigator_diversity.min()}")

# %% Cell 7: Create heatmap of consciousness patterns
# Select top 20 most active navigators
top_navigators = navigator_diversity.nlargest(20).index

# Select top 15 most common patterns
pattern_frequency = (matrix_df > 0).sum(axis=0)
top_patterns = pattern_frequency.nlargest(15).index

# Create subset for visualization
subset_matrix = matrix_df.loc[top_navigators, top_patterns]

# Create heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(subset_matrix, cmap='YlOrRd', cbar_kws={'label': 'Indicator Count'})
plt.title('Consciousness Pattern Heatmap (Top Navigators x Top Patterns)')
plt.xlabel('Consciousness Pattern')
plt.ylabel('Navigator')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# %% Cell 8: Load and analyze overlaps
overlaps_df = pd.read_csv(cluster_dir / "navigator_overlaps.csv")
print(f"Navigators with multiple clusters: {len(overlaps_df)}")

# Overlap statistics
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].hist(overlaps_df['cluster_count'], bins=15, edgecolor='black')
axes[0].set_xlabel('Number of Clusters')
axes[0].set_ylabel('Number of Navigators')
axes[0].set_title('Cluster Participation Distribution')

axes[1].hist(overlaps_df['domain_count'], bins=10, edgecolor='black', color='green')
axes[1].set_xlabel('Number of Domains')
axes[1].set_ylabel('Number of Navigators')
axes[1].set_title('Cross-Domain Navigator Distribution')

axes[2].hist(overlaps_df['type_count'], bins=5, edgecolor='black', color='purple')
axes[2].set_xlabel('Number of Consciousness Types')
axes[2].set_ylabel('Number of Navigators')
axes[2].set_title('Consciousness Type Diversity')

plt.tight_layout()
plt.show()

# %% Cell 9: Analyze specific cluster tweets
# Choose a cluster to analyze
cluster_to_analyze = "healthcare_pattern"  # Change this to any cluster_id

cluster_file = cluster_dir / f"{cluster_to_analyze}_tweets.csv"
if cluster_file.exists():
    tweets_df = pd.read_csv(cluster_file)
    print(f"Analyzing cluster: {cluster_to_analyze}")
    print(f"Total tweets: {len(tweets_df)}")
    print(f"Unique navigators: {tweets_df['navigator_id'].nunique()}")

    # Intensity distribution
    plt.figure(figsize=(10, 4))
    plt.hist(tweets_df['intensity'], bins=20, edgecolor='black')
    plt.xlabel('Intensity Score')
    plt.ylabel('Number of Tweets')
    plt.title(f'Intensity Distribution for {cluster_to_analyze}')
    plt.show()

    # Sample high-intensity tweets
    high_intensity = tweets_df.nlargest(5, 'intensity')[['email', 'intensity', 'content']]
    print("\nHigh intensity examples:")
    for _, row in high_intensity.iterrows():
        print(f"\n[{row['email']}] (Intensity: {row['intensity']:.2f})")
        print(f"  {row['content'][:200]}...")
else:
    print(f"Cluster file not found: {cluster_file}")

# %% Cell 10: Load and analyze communities
communities_file = cluster_dir / "consciousness_communities.csv"
if communities_file.exists():
    communities_df = pd.read_csv(communities_file)
    print(f"Found {len(communities_df)} consciousness communities")

    # Community size distribution
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(communities_df)), communities_df['member_count'])
    plt.xlabel('Community ID')
    plt.ylabel('Number of Members')
    plt.title('Consciousness Community Sizes')
    plt.xticks(range(len(communities_df)),
               [f"C{i+1}" for i in range(len(communities_df))])
    plt.show()

    # Largest community details
    if len(communities_df) > 0:
        largest = communities_df.nlargest(1, 'member_count').iloc[0]
        print(f"\nLargest community:")
        print(f"  Members: {largest['member_count']}")
        print(f"  Shared patterns: {largest['shared_pattern_count']}")
        print(f"  Pattern examples: {largest['shared_patterns'][:100]}...")

# %% Cell 11: Cross-domain analysis
# Find navigators active in multiple domains
if 'domains' in overlaps_df.columns:
    # Parse domains
    overlaps_df['domain_list'] = overlaps_df['domains'].str.split('|')

    # Find cross-domain patterns
    cross_domain = overlaps_df[overlaps_df['domain_count'] > 1]

    print(f"Cross-domain navigators: {len(cross_domain)}")
    print(f"Average domains per cross-domain navigator: {cross_domain['domain_count'].mean():.1f}")

    # Common domain combinations
    from collections import Counter
    domain_combos = Counter()
    for domains in cross_domain['domain_list']:
        if domains:
            combo = tuple(sorted(domains))
            domain_combos[combo] += 1

    print("\nMost common domain combinations:")
    for combo, count in domain_combos.most_common(10):
        print(f"  {' + '.join(combo)}: {count} navigators")

# %% Cell 12: Export summary statistics
summary = {
    'total_clusters': len(clusters_df),
    'total_navigators': matrix_df.shape[0],
    'total_patterns': matrix_df.shape[1],
    'avg_patterns_per_navigator': navigator_diversity.mean(),
    'most_diverse_navigator': {
        'name': navigator_diversity.idxmax(),
        'pattern_count': navigator_diversity.max()
    },
    'largest_cluster': {
        'id': clusters_df.nlargest(1, 'indicator_count')['cluster_id'].iloc[0],
        'size': clusters_df['indicator_count'].max()
    },
    'cross_domain_navigators': len(overlaps_df[overlaps_df['domain_count'] > 1]) if 'domain_count' in overlaps_df.columns else 0,
    'communities_found': len(communities_df) if communities_file.exists() else 0
}

print("\n" + "="*50)
print("CONSCIOUSNESS CLUSTER ANALYSIS SUMMARY")
print("="*50)
for key, value in summary.items():
    if isinstance(value, dict):
        print(f"{key}:")
        for k, v in value.items():
            print(f"  {k}: {v}")
    else:
        print(f"{key}: {value}")

# Save summary
with open(cluster_dir / 'analysis_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print(f"\nSummary saved to {cluster_dir / 'analysis_summary.json'}")