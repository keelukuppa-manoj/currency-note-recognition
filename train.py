# src/train.py
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

# Paths (edit if needed)
train_dir = "data/train"
val_dir = "data/val"
output_model = "model.h5"

img_size = (128, 128)
batch_size = 32
num_epochs = 15
learning_rate = 1e-4

# Data generators
train_gen = ImageDataGenerator(rescale=1./255,
                               rotation_range=15,
                               width_shift_range=0.1,
                               height_shift_range=0.1,
                               zoom_range=0.1,
                               horizontal_flip=True)

val_gen = ImageDataGenerator(rescale=1./255)

train_flow = train_gen.flow_from_directory(train_dir,
                                           target_size=img_size,
                                           batch_size=batch_size,
                                           class_mode='categorical')

val_flow = val_gen.flow_from_directory(val_dir,
                                       target_size=img_size,
                                       batch_size=batch_size,
                                       class_mode='categorical')

num_classes = train_flow.num_classes

# Simple CNN
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(img_size[0], img_size[1], 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.4),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer=Adam(learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

checkpoint = ModelCheckpoint(output_model, monitor='val_accuracy', save_best_only=True, verbose=1)

model.fit(train_flow, validation_data=val_flow, epochs=num_epochs, callbacks=[checkpoint])

print("Saved model to", output_model)
