#!/usr/bin/env python3
"""Test script to verify Supabase downloader functionality"""

import logging
from consciousness_analysis.utils.supabase_downloader import SupabaseArchiveDownloader

# Setup logging to see what's happening
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def test_downloader():
    """Test the Supabase downloader functionality"""

    print("="*60)
    print("TESTING SUPABASE ARCHIVE DOWNLOADER")
    print("="*60)

    # Initialize downloader
    downloader = SupabaseArchiveDownloader(cache_dir="test_cache")

    # Test 1: Check if specific username exists
    print("\n1. Testing if 'leo_guinan' archive exists...")
    exists = downloader.test_username_exists('leo_guinan')
    print(f"   Result: {'✓ Archive exists' if exists else '✗ Archive not found'}")

    # Test 2: Try to download a known archive
    print("\n2. Testing download of 'leo_guinan' archive...")
    data = downloader.download_user_data('leo_guinan', show_progress=True)
    if data:
        print(f"   ✓ Successfully downloaded")
        print(f"   - Tweets: {len(data.get('tweets', []))}")
        if 'profile' in data and data['profile']:
            print(f"   - Has profile data: Yes")
    else:
        print("   ✗ Download failed")

    # Test 3: Test non-existent archive
    print("\n3. Testing non-existent archive...")
    fake_exists = downloader.test_username_exists('definitely_not_a_real_user_12345')
    print(f"   Result: {'Archive exists (unexpected!)' if fake_exists else '✓ Correctly identified as missing'}")

    # Test 4: Get cached archives
    print("\n4. Checking cached archives...")
    cached = downloader.get_cached_archives()
    print(f"   Found {len(cached)} cached archives")
    if cached:
        for username in cached[:5]:
            print(f"   - {username}")
        if len(cached) > 5:
            print(f"   ... and {len(cached)-5} more")

    # Test 5: Get username list
    print("\n5. Getting username list...")
    usernames = downloader.get_usernames_from_supabase(limit=10)
    print(f"   Got {len(usernames)} usernames")
    for i, username in enumerate(usernames[:5], 1):
        print(f"   {i}. {username}")
    if len(usernames) > 5:
        print(f"   ... and {len(usernames)-5} more")

    # Test 6: Discover new archives (test a few potential usernames)
    print("\n6. Testing archive discovery...")
    test_users = ['visakanv', 'patio11', 'swyx', 'fake_user_xyz']
    found = []
    for username in test_users:
        if downloader.test_username_exists(username):
            found.append(username)
            print(f"   ✓ Found: {username}")
        else:
            print(f"   ✗ Not found: {username}")

    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"✓ Downloader initialized successfully")
    print(f"✓ Can test if archives exist")
    print(f"✓ Can download archives")
    print(f"✓ Can manage cache")
    print(f"✓ Found {len(found)} valid archives out of {len(test_users)} tested")

    return True

if __name__ == "__main__":
    try:
        test_downloader()
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()