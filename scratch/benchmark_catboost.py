import time
import numpy as np
from catboost import CatBoostClassifier

# Generate dummy 14,000 x 200 float32 data
X_sample = np.random.randn(14000, 200).astype(np.float32)
y_sample = np.random.randint(0, 2, size=14000).astype(np.int8)

print("Benchmarking CatBoost on CPU with depth 8, thread_count=16...")
t0 = time.time()
clf = CatBoostClassifier(iterations=300, learning_rate=0.04, depth=8, thread_count=16, verbose=100)
clf.fit(X_sample, y_sample)
t1 = time.time()
print(f"300 iterations completed in {t1 - t0:.2f} seconds!")
