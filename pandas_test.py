import pandas as pd
import numpy as np
df = pd.read_excel('orders.xlsx')
for _, row in df.iterrows():
    print(f"order id: {row['id']}")
    print(f"order a: {row['a']}")
    print(f"order b: {row['b']}")
    print(f"order c: {row['c']}")
    print(f"order deadline: {pd.to_datetime(row['deadline']).timestamp()}")
    print("=================================")

orders = []
for _, row in df.iterrows():
    order = {
        'order_id': int(row['id']),
        'components': np.array([int(row['a']), int(row['b']), int(row['c'])]),
        'deadline': pd.to_datetime(row['deadline']).timestamp()
    }
    orders.append(order)

print(len(orders))
print(len(orders[5:]))