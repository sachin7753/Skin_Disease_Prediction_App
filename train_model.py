import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow as tf

# Set paths
IMAGE_DIR_1 = "HAM10000_images_part_1"
IMAGE_DIR_2 = "HAM10000_images_part_2"
IMAGE_DIR = [IMAGE_DIR_1, IMAGE_DIR_2]

# Load metadata
df = pd.read_csv("HAM10000_metadata.csv")
df['image_id'] = df['image_id'] + ".jpg"
df['path'] = df['image_id'].apply(
    lambda x: os.path.join(IMAGE_DIR_1, x) if os.path.exists(os.path.join(IMAGE_DIR_1, x))
    else os.path.join(IMAGE_DIR_2, x)
)

# Use only images that exist
df = df[df['path'].apply(os.path.exists)]

# Encode labels
df['label'] = df['dx'].astype(str)
class_names = sorted(df['label'].unique())
class_indices = {name: idx for idx, name in enumerate(class_names)}
df['label_idx'] = df['label'].map(class_indices)

# Split dataset
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)

# Compute class weights
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_df['label_idx']),
    y=train_df['label_idx']
)
class_weights = dict(enumerate(class_weights))

# Image data generators
train_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True, rotation_range=10, zoom_range=0.1)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(
    train_df,
    x_col='path',
    y_col='label',
    target_size=(224, 224),
    class_mode='categorical',
    batch_size=32
)

val_generator = val_datagen.flow_from_dataframe(
    val_df,
    x_col='path',
    y_col='label',
    target_size=(224, 224),
    class_mode='categorical',
    batch_size=32
)

# Load model
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = True  # Fine-tune the entire model

x = GlobalAveragePooling2D()(base_model.output)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)
output = Dense(len(class_names), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

# Train
model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=30,
    callbacks=[early_stop, reduce_lr],
    class_weight=class_weights
)

# Save model
model.save("skin_model.h5")
