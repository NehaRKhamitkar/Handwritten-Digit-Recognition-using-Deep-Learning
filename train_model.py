"""
train_model.py
---------------
Trains a CNN on the MNIST dataset for handwritten digit recognition
and saves it as 'digit_model.h5'.

Run this ONCE locally (or in Colab) before deploying the Streamlit app:
    python train_model.py

Requires: tensorflow, numpy
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models


def build_model():
    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),

        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def get_augmenter():
    # Small augmentation pipeline -> makes the model more robust to
    # off-center / slightly rotated digits, which matters a LOT for
    # webcam and canvas input (real users don't write like MNIST).
    return tf.keras.Sequential([
        layers.RandomRotation(0.08),
        layers.RandomTranslation(0.08, 0.08),
        layers.RandomZoom(0.08),
    ])


def main():
    print("Loading MNIST...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    augmenter = get_augmenter()

    model = build_model()
    model.summary()

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_ds = train_ds.shuffle(10000).batch(128)
    train_ds = train_ds.map(lambda x, y: (augmenter(x, training=True), y))
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(128)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=4, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=2
        ),
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=25,
        callbacks=callbacks
    )

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nFinal test accuracy: {test_acc:.4f}")

    model.save('digit_model.h5')
    print("Model saved as digit_model.h5")


if __name__ == '__main__':
    main()
