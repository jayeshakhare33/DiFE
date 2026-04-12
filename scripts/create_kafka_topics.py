#!/usr/bin/env python3
"""
Create Kafka Topics
Python version of the batch script
"""
import subprocess
import sys
import time

def create_kafka_topic(topic_name, partitions, replication_factor=1):
    """Create a Kafka topic"""
    cmd = [
        'docker', 'exec', 'fraud-detection-kafka',
        'kafka-topics',
        '--create',
        '--topic', topic_name,
        '--bootstrap-server', 'localhost:9092',
        '--partitions', str(partitions),
        '--replication-factor', str(replication_factor),
        '--if-not-exists'
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"✅ Created topic: {topic_name}")
        return True
    except subprocess.CalledProcessError as e:
        if "already exists" in e.stderr.lower() or "TopicExistsException" in e.stderr:
            print(f"ℹ️  Topic already exists: {topic_name}")
            return True
        else:
            print(f"❌ Failed to create topic {topic_name}: {e.stderr}")
            return False

def list_topics():
    """List all Kafka topics"""
    cmd = [
        'docker', 'exec', 'fraud-detection-kafka',
        'kafka-topics',
        '--list',
        '--bootstrap-server', 'localhost:9092'
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        topics = result.stdout.strip().split('\n')
        return topics
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to list topics: {e.stderr}")
        return []

def main():
    print("=" * 60)
    print("Creating Kafka Topics")
    print("=" * 60)
    print()
    
    # Wait for Kafka to be ready
    print("Waiting for Kafka to be ready...")
    time.sleep(5)
    
    # Check if Kafka is accessible
    try:
        check_cmd = [
            'docker', 'exec', 'fraud-detection-kafka',
            'kafka-broker-api-versions',
            '--bootstrap-server', 'localhost:9092'
        ]
        subprocess.run(check_cmd, capture_output=True, check=True)
        print("✅ Kafka is ready!")
    except subprocess.CalledProcessError:
        print("⚠️  Kafka might not be ready yet. Continuing anyway...")
    
    print()
    
    # Topics to create
    topics = [
        ('transactions-raw', 6),
        ('graph-updates', 3),
        ('feature-extraction', 4),
        ('features-computed', 4),
        ('features-stored', 1),
    ]
    
    success_count = 0
    for topic_name, partitions in topics:
        if create_kafka_topic(topic_name, partitions):
            success_count += 1
        time.sleep(1)  # Small delay between creations
    
    print()
    print("=" * 60)
    print(f"Created {success_count}/{len(topics)} topics")
    print("=" * 60)
    print()
    
    # List all topics
    print("Listing all topics:")
    print("-" * 60)
    existing_topics = list_topics()
    if existing_topics:
        for topic in sorted(existing_topics):
            print(f"  - {topic}")
    else:
        print("  (No topics found)")
    
    print()
    print("Done!")
    return 0 if success_count == len(topics) else 1

if __name__ == '__main__':
    sys.exit(main())

