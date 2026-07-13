import os
import matplotlib.pyplot as plt

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)

from dataset import load_dataset
from model import build_model
from config import *



os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)



print("=" * 50)
print("Loading Dataset...")
print("=" * 50)

train_generator, validation_generator = load_dataset()

print("\nDataset Loaded Successfully!\n")

print(f"Training Images   : {train_generator.samples}")
print(f"Validation Images : {validation_generator.samples}")

print("\nClass Mapping")

print(train_generator.class_indices)



print("\nBuilding EfficientNetB0 Model...\n")

model = build_model()

model.summary()



checkpoint = ModelCheckpoint(
    filepath=BEST_MODEL_PATH,
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



print("\nTraining Started...\n")

history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=EPOCHS,
    callbacks=[
        checkpoint,
        early_stop,
        reduce_lr
    ]
)

# ==========================================
# Save Final Model
# ==========================================

model.save(FINAL_MODEL_PATH)

print("\nFinal Model Saved Successfully!")

