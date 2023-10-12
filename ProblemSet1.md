# Task 1: Load MNIST and Show Montage
```python
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

(x_train, _), (_, _) = mnist.load_data()

plt.figure(figsize=(10, 10))
montage = np.zeros((28 * 5, 28 * 5))

for i in range(5):
    for j in range(5):
        montage[i * 28: (i + 1) * 28, j * 28: (j + 1) * 28] = x_train[np.random.randint(0, x_train.shape[0])]

plt.imshow(montage, cmap='gray')
plt.axis('off')
plt.title('Montage of MNIST Images')
plt.show()
```python

# Task 2: Run Random y=mx Model on MNIST
```python
import numpy as np

m = np.random.rand()
x_train_flat = x_train.reshape(x_train.shape[0], -1)
predictions = m * x_train_flat

print(f"Random Slope (m): {m}")
print(f"Sample Predictions: {predictions[:5]}")
```python

# Task 3: Train random walk model to at least 75%
```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

x_train_flat = x_train.reshape(x_train.shape[0], -1)
y_random_walk = np.random.randint(0, 2, size=x_train_flat.shape[0])

x_train_rw, x_test_rw, y_train_rw, y_test_rw = train_test_split(x_train_flat, y_random_walk, test_size=0.2, random_state=42)

# Use a simple random walk model
class RandomWalkModel:
    def fit(self, x, y):
        pass  # No training needed for a random walk model

    def predict(self, x):
        return np.random.randint(0, 2, size=x.shape[0])

# Create and train the random walk model
rw_model = RandomWalkModel()
rw_model.fit(x_train_rw, y_train_rw)

# Make predictions and evaluate accuracy
y_pred_rw = rw_model.predict(x_test_rw)

accuracy = accuracy_score(y_test_rw, y_pred_rw)
print(f"Accuracy of Random Walk Model: {accuracy * 100:.2f}%")
```python
