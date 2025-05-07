


import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models

# Load the Fashion MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# Normalize pixel values to 0–1
x_train, x_test = x_train / 255.0, x_test / 255.0

# Add channel dimension (28,28) → (28,28,1) for CNN compatibility
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

# Print dataset shape to confirm 60k train, 10k test
print(f"Training data shape: {x_train.shape}")
print(f"Testing data shape: {x_test.shape}")

# Build CNN model
model = models.Sequential([
    #32 number of filter and 3,3 is dimention of filter
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),  # Adding Dropout for regularization
    layers.Dense(10, activation='softmax')  # 10 categories softmax converts numbers to probabilities greater no greater probability
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model (80% training, 20% validation)
model.fit(x_train, y_train, epochs=5, validation_split=0.2)

# Evaluate the model on test data
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")

# Make a prediction on the first test image
prediction = model.predict(x_test[:1])
predicted_label = np.argmax(prediction[0])  #output possiblilities of images is stored in array choose greates ones index as label
print(f"Predicted Label: {predicted_label}")
print(f"Actual Label: {y_test[0]}")

# Visualize the test image
plt.imshow(x_test[0].reshape(28, 28), cmap=plt.cm.gray)
plt.title(f"Predicted: {predicted_label}, Actual: {y_test[0]}")
plt.axis("off")
plt.show()

#Input Image → Convolution → ReLU → Pooling → (Repeat) → Flatten → Fully Connected → Output





