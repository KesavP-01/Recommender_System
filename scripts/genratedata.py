import pandas as pd
import numpy as np
import random


def generate_data(n_users = 2000, n_items=500, n_iteractions=20000):
    users = [f'user_{i}' for i in range(n_users)]
    items = [f'item_{i}' for i in range(n_items)]
    data=[]

    for _ in range(n_iteractions):
        user = random.choice(users)
        item = random.choice(items)
        like = np.random.choice([0,1], p=[0.8, 0.2])
        data.append([user, item, like])

    df = pd.DataFrame(data, columns=['user_id', 'item_id', 'like'])
    df.to_csv('data/interactions.csv', index=False)

if __name__ == "__main__":
    generate_data()