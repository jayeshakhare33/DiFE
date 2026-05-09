import pandas as pd
import numpy as np
import time

def generate_dummy_data():
    np.random.seed(42)
    n_transactions = 100
    
    # Generate random data
    data = {
        'transaction_id': [f'tx_{i}' for i in range(n_transactions)],
        'sender_id': np.random.randint(1, 15, n_transactions),
        'receiver_id': np.random.randint(1, 15, n_transactions),
        'amount': np.round(np.random.uniform(10.0, 5000.0, n_transactions), 2),
        'timestamp': [time.time() - np.random.randint(0, 1000000) for _ in range(n_transactions)]
    }
    
    df = pd.DataFrame(data)
    df.to_csv('test_transactions.csv', index=False)
    print("Created test_transactions.csv with 100 rows.")

if __name__ == "__main__":
    generate_dummy_data()
