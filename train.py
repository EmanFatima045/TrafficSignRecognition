import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Image size
IMG_SIZE = 32
NUM_CLASSES = 43

# Your dataset path
DATASET_PATH = r"D:\Traffic Sign Recognition\DataSet\trafficSign"

data = []
labels = []

# Load images (since they are not in class folders)
for img_name in os.listdir(DATASET_PATH):
    if img_name.endswith(".png") or img_name.endswith(".jpg"):
        img_path = os.path.join(DATASET_PATH, img_name)

        img = cv2.imread(img_path)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        data.append(img)

        # Extract label from filename: "14.png" -> 14
        label = int(img_name.split(".")[0])
        labels.append(label)

# Convert to numpy arrays
data = np.array(data) / 255.0
labels = np.array(labels)

# One-hot encode labels
labels = to_categorical(labels, NUM_CLASSES)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=42
)

# Build CNN model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation='relu'),
    Dense(NUM_CLASSES, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train model
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Save model
model.save("traffic_sign_model.h5")

print(" Model trained and saved successfully!")
