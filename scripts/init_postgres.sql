-- PostgreSQL Database Initialization Script
-- This script creates the database schema for fraud detection

-- Create database (if not exists)
-- Note: Run this manually if database doesn't exist:
-- CREATE DATABASE fraud_detection;

-- Connect to database (run manually)
-- \c fraud_detection;

-- Transactions table
CREATE TABLE IF NOT EXISTS transactions (
    transaction_id BIGSERIAL PRIMARY KEY,
    sender_id VARCHAR(50) NOT NULL,
    receiver_id VARCHAR(50) NOT NULL,
    amount DECIMAL(15, 2) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    location_id VARCHAR(50),
    device_id VARCHAR(50),
    card_id VARCHAR(50),
    is_fraud BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Users table
CREATE TABLE IF NOT EXISTS users (
    user_id VARCHAR(50) PRIMARY KEY,
    name VARCHAR(255),
    email VARCHAR(255),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Locations table
CREATE TABLE IF NOT EXISTS locations (
    location_id VARCHAR(50) PRIMARY KEY,
    country VARCHAR(100),
    city VARCHAR(100),
    latitude DECIMAL(10, 8),
    longitude DECIMAL(11, 8)
);

-- Devices table
CREATE TABLE IF NOT EXISTS devices (
    device_id VARCHAR(50) PRIMARY KEY,
    device_type VARCHAR(100),
    device_info VARCHAR(255),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Cards table
CREATE TABLE IF NOT EXISTS cards (
    card_id VARCHAR(50) PRIMARY KEY,
    card_type VARCHAR(50),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_transactions_sender ON transactions(sender_id);
CREATE INDEX IF NOT EXISTS idx_transactions_receiver ON transactions(receiver_id);
CREATE INDEX IF NOT EXISTS idx_transactions_timestamp ON transactions(timestamp);
CREATE INDEX IF NOT EXISTS idx_transactions_updated_at ON transactions(updated_at);
CREATE INDEX IF NOT EXISTS idx_transactions_is_fraud ON transactions(is_fraud);
CREATE INDEX IF NOT EXISTS idx_transactions_location ON transactions(location_id);
CREATE INDEX IF NOT EXISTS idx_transactions_device ON transactions(device_id);
CREATE INDEX IF NOT EXISTS idx_transactions_card ON transactions(card_id);

-- Insert sample data (optional - for testing)
INSERT INTO users (user_id, name, email) VALUES
    ('user_001', 'John Doe', 'john@example.com'),
    ('user_002', 'Jane Smith', 'jane@example.com'),
    ('user_003', 'Bob Johnson', 'bob@example.com')
ON CONFLICT (user_id) DO NOTHING;

INSERT INTO locations (location_id, country, city, latitude, longitude) VALUES
    ('loc_001', 'USA', 'New York', 40.7128, -74.0060),
    ('loc_002', 'USA', 'Los Angeles', 34.0522, -118.2437),
    ('loc_003', 'USA', 'Chicago', 41.8781, -87.6298)
ON CONFLICT (location_id) DO NOTHING;

INSERT INTO devices (device_id, device_type, device_info) VALUES
    ('dev_001', 'mobile', 'iPhone 13'),
    ('dev_002', 'desktop', 'Windows PC'),
    ('dev_003', 'mobile', 'Android Phone')
ON CONFLICT (device_id) DO NOTHING;

INSERT INTO cards (card_id, card_type) VALUES
    ('card_001', 'credit'),
    ('card_002', 'debit'),
    ('card_003', 'credit')
ON CONFLICT (card_id) DO NOTHING;

-- Sample transactions
INSERT INTO transactions (sender_id, receiver_id, amount, timestamp, location_id, device_id, card_id, is_fraud) VALUES
    ('user_001', 'user_002', 1500.50, NOW() - INTERVAL '1 day', 'loc_001', 'dev_001', 'card_001', FALSE),
    ('user_002', 'user_001', 750.25, NOW() - INTERVAL '12 hours', 'loc_002', 'dev_002', 'card_002', FALSE),
    ('user_001', 'user_003', 2000.00, NOW() - INTERVAL '6 hours', 'loc_001', 'dev_001', 'card_001', FALSE),
    ('user_003', 'user_002', 500.00, NOW() - INTERVAL '3 hours', 'loc_003', 'dev_003', 'card_003', TRUE),
    ('user_002', 'user_003', 1200.75, NOW() - INTERVAL '1 hour', 'loc_002', 'dev_002', 'card_002', FALSE)
ON CONFLICT DO NOTHING;

-- Create a function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create trigger to auto-update updated_at
DROP TRIGGER IF EXISTS update_transactions_updated_at ON transactions;
CREATE TRIGGER update_transactions_updated_at
    BEFORE UPDATE ON transactions
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Grant permissions (adjust as needed)
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO postgres;
-- GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO postgres;

-- Display summary
SELECT 
    'Database initialized successfully!' as status,
    (SELECT COUNT(*) FROM users) as users_count,
    (SELECT COUNT(*) FROM locations) as locations_count,
    (SELECT COUNT(*) FROM transactions) as transactions_count,
    (SELECT COUNT(*) FROM transactions WHERE is_fraud = TRUE) as fraud_count;

