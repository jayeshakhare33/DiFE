#!/usr/bin/env python3
"""
Test Redis Connection
"""
import redis
import sys
import os

def test_redis():
    """Test Redis connection and basic operations"""
    try:
        # Get connection parameters
        host = os.getenv('REDIS_HOST', 'localhost')
        port = int(os.getenv('REDIS_PORT', '6379'))
        password = os.getenv('REDIS_PASSWORD', None)
        db = int(os.getenv('REDIS_DB', '0'))
        
        print(f"Connecting to Redis at {host}:{port}...")
        
        # Connect to Redis
        r = redis.Redis(
            host=host,
            port=port,
            password=password,
            db=db,
            decode_responses=True
        )
        
        # Test connection
        r.ping()
        print("✅ Redis connection successful!")
        
        # Test set/get
        r.set('test_key', 'Hello Redis!')
        value = r.get('test_key')
        print(f"✅ Set/Get test successful! Value: {value}")
        
        # Check for feature keys
        feature_keys = r.keys('features:*')
        print(f"✅ Found {len(feature_keys)} feature keys in Redis")
        
        if feature_keys:
            print("   Sample keys:")
            for key in sorted(feature_keys)[:5]:
                print(f"   - {key}")
        
        # Check metadata
        node_metadata = r.get('features:metadata:node')
        edge_metadata = r.get('features:metadata:edge')
        
        if node_metadata:
            print(f"✅ Node features metadata exists")
        if edge_metadata:
            print(f"✅ Edge features metadata exists")
        
        # Clean up test key
        r.delete('test_key')
        print("✅ Cleanup successful!")
        
        print("\n✅ All Redis tests passed!")
        return True
        
    except redis.ConnectionError as e:
        print(f"❌ Connection Error: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure Redis is running (check docker-compose ps)")
        print("2. Check connection parameters")
        print("3. Verify Redis is accessible at localhost:6379")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = test_redis()
    sys.exit(0 if success else 1)

