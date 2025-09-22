"""
Example usage for Jupyter notebooks
Copy this code into your notebook cells
"""

# %% Cell 1: Setup and imports
import sys
sys.path.append('.')  # Add current directory to path

from consciousness_analysis.utils.supabase_downloader import SupabaseArchiveDownloader
from consciousness_analysis.core.database import DatabaseManager
from consciousness_analysis.processors.community_archive import CommunityArchiveProcessor
from consciousness_analysis.utils.reporting import generate_navigator_report

# %% Cell 2: Initialize downloader
downloader = SupabaseArchiveDownloader(cache_dir="archive_cache")

# %% Cell 3: Get list of available usernames from Supabase
usernames = downloader.get_usernames_from_supabase(limit=10)
print(f"Found {len(usernames)} usernames:")
for username in usernames[:5]:
    print(f"  - {username}")
print(f"  ... and {len(usernames)-5} more")

# %% Cell 4: Download a single user
username = 'leo_guinan'
data = downloader.download_user_data(username, show_progress=True)

if data:
    print(f"\nDownloaded archive for {username}")
    print(f"  - Tweets: {len(data.get('tweets', []))}")
    if 'profile' in data and data['profile']:
        profile = data['profile'][0].get('profile', {})
        bio = profile.get('description', {}).get('bio', 'No bio')
        print(f"  - Bio: {bio[:100]}...")

# %% Cell 5: Download multiple users with progress tracking
# This will show nested progress bars in Jupyter
selected_users = usernames[:5]  # Process first 5 users
archives = downloader.download_all_users(
    usernames=selected_users,
    show_progress=True
)

print(f"\nSuccessfully downloaded {len(archives)} archives")

# %% Cell 6: Analyze downloaded archives
db_manager = DatabaseManager("bottega_notebook.db")
processor = CommunityArchiveProcessor(
    db_manager=db_manager,
    cache_dir="archive_cache"
)

# Process each downloaded user
for username in archives.keys():
    print(f"\nAnalyzing {username}...")
    navigator_id = processor.process_user(username)
    if navigator_id:
        print(f"  ✓ Created navigator profile: {navigator_id}")

        # Get some quick stats
        stats = db_manager.execute_query("""
            SELECT
                SUM(estimated_hours) as total_hours,
                COUNT(*) as incident_count
            FROM time_violence_incidents
            WHERE navigator_id = ?
        """, (navigator_id,))

        if stats:
            row = stats[0]
            print(f"  - Time Violence: {row['total_hours']:.1f} hours in {row['incident_count']} incidents")

# %% Cell 7: Generate summary report
from consciousness_analysis.utils.reporting import generate_community_report

report = generate_community_report(db_manager)
print(report)

# %% Cell 8: Query specific patterns
# Find top consciousness patterns
patterns = db_manager.execute_query("""
    SELECT
        domain,
        pattern_description,
        occurrence_count,
        navigators_demonstrating
    FROM community_patterns
    WHERE occurrence_count > 1
    ORDER BY occurrence_count DESC
    LIMIT 10
""")

if patterns:
    print("\nTop Recurring Patterns:")
    for pattern in patterns:
        print(f"  [{pattern['domain']}] {pattern['pattern_description']}")
        print(f"    Found in {pattern['occurrence_count']} navigators")

# %% Cell 9: Export results for further analysis
import json
import pandas as pd

# Export navigator data to DataFrame
navigators = db_manager.execute_query("""
    SELECT
        n.navigator_id,
        n.email,
        cc.consciousness_stage,
        cc.complexity_scale,
        cc.fragmentation_resistance,
        cc.consciousness_sharing_ability,
        cc.pattern_recognition_depth
    FROM navigators n
    JOIN consciousness_capacity cc ON n.navigator_id = cc.navigator_id
""")

if navigators:
    df = pd.DataFrame([dict(row) for row in navigators])
    df['username'] = df['email'].str.replace('@twitter.import', '')

    print("Navigator Metrics Summary:")
    print(df[['username', 'consciousness_stage', 'fragmentation_resistance']].head())

    # Save to CSV for further analysis
    df.to_csv('navigator_analysis.csv', index=False)
    print("\nExported to navigator_analysis.csv")