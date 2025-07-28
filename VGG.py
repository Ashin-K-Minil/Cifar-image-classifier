import json
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.applications import VGG16 # type: ignore
from tensorflow.keras.callbacks import ModelCheckpoint # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.layers import Dense, Flatten, Dropout, LeakyReLU, BatchNormalization # type: ignore

# Loading the saved preprocessed data
data = np.load('cifar10_preprocessed.npz')
X_train = data['X_train']
X_test = data['X_test']
y_train = data['y_train']
y_test = data['y_test']

# Loading the pretrained model
base_model = VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
base_model.trainable = False

# Building custom layer at the output
x = Flatten()(base_model.output)
x = Dense(512)(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha= 0.01)(x)
x = Dropout(0.3)(x)
output = Dense(10, activation='softmax')(x)

model = Model(inputs = base_model.input, outputs = output)

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=['accuracy'])

# Saving the model
checkpoint = ModelCheckpoint(
    filepath='vgg16_cifar10_best_model.h5',
    monitor='val_loss',
    save_best_only=True,
    save_weights_only=False,
    mode='min',
    verbose=1
)

# Image augmentation
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)
train_generator = datagen.flow(X_train, y_train, batch_size=32)

# Training the model
history = model.fit(
    train_generator,
    steps_per_epoch=len(X_train)//32,
    epochs=35,
    validation_data=(X_test, y_test),
    callbacks=[checkpoint]
)

# Save the history - model training
with open('training_history.json', 'w') as f:
    json.dump(history.history, f)