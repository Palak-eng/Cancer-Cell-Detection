import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_path = "Cancer_Dataset/train"
test_path = "Cancer_Dataset/test"

print("Train path:", train_path)
print("Test path :", test_path)

print("\nChecking if folders exist...\n")

print("Train Folder Exists:", os.path.exists(train_path))
print("Test Folder Exists :", os.path.exists(test_path))

train_datagen = ImageDataGenerator(
    # NOTE: no rescale=1./255 here.
    # EfficientNetB0 has its own normalization built into the model graph
    # (it expects raw 0-255 pixel values). Rescaling here AND inside the
    # model was the double-normalization bug causing ~50% confidence.
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

validation_datagen = ImageDataGenerator(
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    directory=train_path,
    target_size=(224,224),
    batch_size=32,
    class_mode='binary',
    subset='training',
    shuffle=True
)

validation_generator = validation_datagen.flow_from_directory(
    directory=train_path,
    target_size=(224,224),
    batch_size=32,
    class_mode='binary',
    subset='validation',
    shuffle=False
)

print("\nDataset Loaded Successfully!\n")

print("Training Images   :", train_generator.samples)
print("Validation Images :", validation_generator.samples)

print("\nClasses")

print(train_generator.class_indices)

from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout

base_model = EfficientNetB0(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)
base_model.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.summary()

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

os.makedirs("models", exist_ok=True)

checkpoint = ModelCheckpoint(
    filepath="models/best_model.keras",
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.2,
    patience=3,
    min_lr=1e-6,
    verbose=1
)
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=15,
    callbacks=[
        checkpoint,
        early_stop,
        reduce_lr
    ]
)

model.save("models/final_model.keras")

print("\nModel Saved Successfully!")