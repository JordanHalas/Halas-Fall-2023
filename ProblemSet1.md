# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

# Load MNIST dataset
(x_train, _), (_, _) = mnist.load_data()

# Display a montage of images
plt.figure(figsize=(10, 10))
montage = np.zeros((28 * 5, 28 * 5))

for i in range(5):
    for j in range(5):
        montage[i * 28: (i + 1) * 28, j * 28: (j + 1) * 28] = x_train[np.random.randint(0, x_train.shape[0])]

plt.imshow(montage, cmap='gray')
plt.axis('off')
plt.title('Montage of MNIST Images')
plt.show()




# Assuming you want a random slope (m) for y=mx model
m = np.random.rand()

# Flatten images for linear regression
x_train_flat = x_train.reshape(x_train.shape[0], -1)

# Generate random predictions using y=mx model
predictions = m * x_train_flat

# Display results (you might want to compare predictions with actual labels)
print(f"Random Slope (m): {m}")
print(f"Sample Predictions: {predictions[:5]}")



# Assuming you want a simple random walk model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Flatten images for random walk model
x_train_flat = x_train.reshape(x_train.shape[0], -1)

# Generate random labels for random walk
y_random_walk = np.random.randint(0, 2, size=x_train_flat.shape[0])

# Split data into training and testing sets
x_train_rw, x_test_rw, y_train_rw, y_test_rw = train_test_split(x_train_flat, y_random_walk, test_size=0.2, random_state=42)

# Train Random Forest classifier
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(x_train_rw, y_train_rw)

# Make predictions on the test set
y_pred_rw = rf_classifier.predict(x_test_rw)

# Calculate accuracy
accuracy = accuracy_score(y_test_rw, y_pred_rw)
print(f"Accuracy of Random Walk Model: {accuracy * 100:.2f}%")
