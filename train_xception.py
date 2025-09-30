# train_xception.py

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Step 1: Dataset Path and Validation
dataset_path = 'C:/ai_image_detector/deepfake_dataset'

# Verify dataset structure
def validate_dataset_structure(dataset_path):
    real_path = os.path.join(dataset_path, 'real')
    fake_path = os.path.join(dataset_path, 'fake')
    
    if not os.path.exists(real_path) or not os.path.exists(fake_path):
        raise ValueError("Dataset must contain 'real' and 'fake' subdirectories")
    
    real_files = os.listdir(real_path)
    fake_files = os.listdir(fake_path)
    
    print(f"Found {len(real_files)} real images and {len(fake_files)} fake images")
    
    # Sample check for potential misclassifications
    print("\nSample checking real images for anomalies...")
    for f in real_files[:5]:
        if 'fake' in f.lower() or 'generated' in f.lower():
            print(f"⚠️ Warning: Suspicious filename in real folder: {f}")
    
    print("\nSample checking fake images for anomalies...")
    for f in fake_files[:5]:
        if 'real' in f.lower() or 'authentic' in f.lower():
            print(f"⚠️ Warning: Suspicious filename in fake folder: {f}")

validate_dataset_structure(dataset_path)

# Step 2: Enhanced Data Preparation
img_size = (299, 299)
batch_size = 32  # Increased batch size for better generalization

# Enhanced Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2,
    brightness_range=[0.8, 1.2]  # Handle different lighting conditions
)

# More rigorous validation generator
val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_gen = train_datagen.flow_from_directory(
    dataset_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='training',
    shuffle=True,
    seed=42  # Fixed seed for reproducibility
)

val_gen = val_datagen.flow_from_directory(
    dataset_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='validation',
    shuffle=False
)

# Step 3: Enhanced Model Architecture
def build_model():
    base_model = Xception(
        weights='imagenet',
        include_top=False,
        input_shape=(299, 299, 3),
        pooling=None
    )
    
    # Freeze initial layers, fine-tune later ones
    for layer in base_model.layers[:50]:
        layer.trainable = False
    for layer in base_model.layers[50:]:
        layer.trainable = True
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    output = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=base_model.input, outputs=output)
    
    optimizer = Adam(learning_rate=1e-4)
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    return model

model = build_model()
model.summary()

# Step 4: Enhanced Training with Callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ModelCheckpoint('best_xception.h5', monitor='val_loss', save_best_only=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)
]

history = model.fit(
    train_gen,
    steps_per_epoch=len(train_gen),
    validation_data=val_gen,
    validation_steps=len(val_gen),
    epochs=25,  # Increased epochs for better convergence
    callbacks=callbacks,
    verbose=1
)

# Step 5: Model Evaluation
print("\nEvaluating model on validation set...")
val_preds = model.predict(val_gen)
val_preds = (val_preds > 0.5).astype(int)  # Convert to binary predictions

print("\nClassification Report:")
print(classification_report(val_gen.classes, val_preds))

print("\nConfusion Matrix:")
cm = confusion_matrix(val_gen.classes, val_preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Real', 'Fake'], 
            yticklabels=['Real', 'Fake'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('confusion_matrix.png')
plt.show()

# Step 6: Save the Final Model
model.save('xception.h5')
print("✅ Final model saved as xception.h5")

# Step 7: Training Visualization
plt.figure(figsize=(18, 6))

# Accuracy plot
plt.subplot(1, 3, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss plot
plt.subplot(1, 3, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Precision-Recall plot
plt.subplot(1, 3, 3)
plt.plot(history.history['precision'], label='Train Precision')
plt.plot(history.history['val_precision'], label='Val Precision')
plt.plot(history.history['recall'], label='Train Recall')
plt.plot(history.history['val_recall'], label='Val Recall')
plt.title('Precision & Recall')
plt.xlabel('Epoch')
plt.ylabel('Score')
plt.legend()

plt.tight_layout()
plt.savefig('training_metrics.png')
plt.show()