#!/usr/bin/env python3
"""
Test Kafka Connection
"""
from kafka import KafkaProducer, KafkaConsumer, KafkaAdminClient
from kafka.errors import KafkaError
import json
import time
import sys
import os

def test_kafka():
    """Test Kafka connection and basic operations"""
    try:
        bootstrap_servers = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092').split(',')
        
        print(f"Connecting to Kafka at {bootstrap_servers}...")
        
        # Test Admin Client
        admin_client = KafkaAdminClient(bootstrap_servers=bootstrap_servers)
        print("✅ Kafka Admin Client connection successful!")
        
        # List topics
        topics = admin_client.list_topics()
        print(f"✅ Found {len(topics)} topics:")
        for topic in sorted(topics):
            print(f"   - {topic}")
        
        admin_client.close()
        
        # Test Producer
        print("\nTesting Producer...")
        producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            request_timeout_ms=5000
        )
        
        test_message = {
            'test': True,
            'message': 'Hello Kafka!',
            'timestamp': time.time()
        }
        
        future = producer.send('test-topic', value=test_message)
        record_metadata = future.get(timeout=10)
        print(f"✅ Producer test successful! Message sent to topic: {record_metadata.topic}, partition: {record_metadata.partition}, offset: {record_metadata.offset}")
        producer.close()
        
        # Test Consumer
        print("\nTesting Consumer...")
        consumer = KafkaConsumer(
            'test-topic',
            bootstrap_servers=bootstrap_servers,
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            auto_offset_reset='earliest',
            consumer_timeout_ms=5000,
            group_id='test-group'
        )
        
        print("✅ Consumer created! Waiting for messages...")
        message_found = False
        for message in consumer:
            print(f"✅ Consumer test successful! Received: {message.value}")
            message_found = True
            break
        
        if not message_found:
            print("⚠️  No messages received (this is OK if topic was just created)")
        
        consumer.close()
        
        print("\n✅ All Kafka tests passed!")
        return True
        
    except KafkaError as e:
        print(f"❌ Kafka Error: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure Kafka is running (check docker-compose ps)")
        print("2. Wait a few seconds for Kafka to fully start")
        print("3. Check if Kafka is accessible at localhost:9092")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = test_kafka()
    sys.exit(0 if success else 1)

