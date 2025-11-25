@echo off
echo ========================================
echo Creating Kafka Topics
echo ========================================
echo.

echo Waiting for Kafka to be ready...
timeout /t 10 /nobreak

echo Creating topic: transactions-raw
docker exec fraud-detection-kafka kafka-topics --create --topic transactions-raw --bootstrap-server localhost:9092 --partitions 6 --replication-factor 1 --if-not-exists

echo Creating topic: graph-updates
docker exec fraud-detection-kafka kafka-topics --create --topic graph-updates --bootstrap-server localhost:9092 --partitions 3 --replication-factor 1 --if-not-exists

echo Creating topic: feature-extraction
docker exec fraud-detection-kafka kafka-topics --create --topic feature-extraction --bootstrap-server localhost:9092 --partitions 4 --replication-factor 1 --if-not-exists

echo Creating topic: features-computed
docker exec fraud-detection-kafka kafka-topics --create --topic features-computed --bootstrap-server localhost:9092 --partitions 4 --replication-factor 1 --if-not-exists

echo Creating topic: features-stored
docker exec fraud-detection-kafka kafka-topics --create --topic features-stored --bootstrap-server localhost:9092 --partitions 1 --replication-factor 1 --if-not-exists

echo.
echo ========================================
echo Listing all topics:
echo ========================================
docker exec fraud-detection-kafka kafka-topics --list --bootstrap-server localhost:9092

echo.
echo Done!
pause

