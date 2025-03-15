import keras
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras._tf_keras.keras.applications import MobileNetV2
from keras._tf_keras.keras.models import Model
from keras._tf_keras.keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D
from keras._tf_keras.keras.optimizers import Adam
from keras._tf_keras.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras._tf_keras.keras.utils import to_categorical
from keras._tf_keras.keras.datasets import cifar10
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import tf2onnx
import onnx
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize images
x_train, x_test = x_train / 255.0, x_test / 255.0

# Convert labels to categorical
num_classes = 10
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2
)
datagen.fit(x_train)

# Load MobileNetV2 (better for small images)
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

# Unfreeze last few layers
for layer in base_model.layers[-20:]:
    layer.trainable = True

# Freeze BatchNorm layers
for layer in base_model.layers:
    if 'batch_normalization' in layer.name:
        layer.trainable = False

# Build Model
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

# Compile model (Lower LR)
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)

# Train Model (Use augmented data, 50 epochs)
history = model.fit(
    datagen.flow(x_train, y_train, batch_size=32),
    validation_data=(x_test, y_test),
    epochs=50,
    batch_size=32,
    callbacks=[early_stop, reduce_lr]
)

# Evaluate Model
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# Print Metrics
print(f"Accuracy: {accuracy_score(y_true, y_pred_classes):.4f}")
print("\nClassification Report:\n", classification_report(y_true, y_pred_classes))

# Confusion Matrix
conf_matrix = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(8,6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
#plt.show()


import os
# Ensure the "models" directory exists
if not os.path.exists("models"):
    os.makedirs("models")
# Save Model in HDF5 format (Alternative to Pickle)
model.save("models/model.h5")



print("Model successfully saved  HDF5 (.h5) at models/!")
