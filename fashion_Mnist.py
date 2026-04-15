# Fashion MNIST Classifier using Neural Network (Simple Complete Code)

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.datasets import fashion_mnist

# Load Fashion MNIST dataset
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Normalize pixel values (0-255 → 0-1)
train_images = train_images / 255.0
test_images = test_images / 255.0

# Class names
class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# Build Neural Network Model
model = Sequential([
    Flatten(input_shape=(28, 28)),      # Convert 28x28 image into 1D vector
    Dense(128, activation='relu'),      # Hidden layer
    Dense(10, activation='softmax')     # Output layer for 10 classes
])

# Compile model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train model
model.fit(train_images, train_labels, epochs=10)

# Evaluate model
test_loss, test_accuracy = model.evaluate(test_images, test_labels)

print("\nTest Accuracy:", test_accuracy)

# Make predictions
predictions = model.predict(test_images)

# Show one sample prediction
sample_index = 0
predicted_label = class_names[predictions[sample_index].argmax()]
actual_label = class_names[test_labels[sample_index]]

print(f"\nPredicted: {predicted_label}")
print(f"Actual: {actual_label}")