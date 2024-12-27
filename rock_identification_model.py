import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import matplotlib.pyplot as plt

# Set dataset path
dataset_path = './dataset'  # Path to your dataset

# Prepare the data with ImageDataGenerator
datagen = ImageDataGenerator(
    rescale=1.0/255,  # Normalize image pixel values to [0, 1]
    validation_split=0.2  # 20% data for validation
)

# Load training data
train_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(64, 64),  # Resize images to 64x64
    batch_size=32,  # Batch size
    class_mode='categorical',  # Multi-class classification
    subset='training'  # Use data for training
)

# Load validation data
val_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(64, 64),  # Resize images to 64x64
    batch_size=32,  # Batch size
    class_mode='categorical',  # Multi-class classification
    subset='validation'  # Use data for validation
)

# Build CNN model
model = Sequential()

# Add convolutional layers
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))  # First Conv2D layer
model.add(MaxPooling2D(pool_size=(2, 2)))  # MaxPooling layer

model.add(Conv2D(64, (3, 3), activation='relu'))  # Second Conv2D layer
model.add(MaxPooling2D(pool_size=(2, 2)))  # MaxPooling layer

# Flatten the data before feeding it into fully connected layers
model.add(Flatten())

# Add fully connected layers
model.add(Dense(128, activation='relu'))  # Dense layer
model.add(Dense(3, activation='softmax'))  # Output layer (3 classes)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(train_data, validation_data=val_data, epochs=10)

# Plot training & validation accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Save the trained model
model.save('rock_classification_model.h5')
