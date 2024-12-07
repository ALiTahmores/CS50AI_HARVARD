import os
import sys
import numpy as np
import tensorflow as tf
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Constants
EPOCHS = 20
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():
    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Load and preprocess data
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE, random_state=42
    )

    # Create model
    model = get_model()

    # Train model with callbacks
    callbacks = get_callbacks()
    model.fit(
        x_train, y_train,
        validation_split=0.2,
        epochs=EPOCHS,
        batch_size=32,
        callbacks=callbacks
    )

    # Evaluate model on test data
    print("\nEvaluating model on test data...")
    model.evaluate(x_test, y_test, verbose=2)

    # Save model if requested
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):
    """
    Load and preprocess image data from directory.
    """
    print(f'Loading images from directory: "{data_dir}"...')
    images, labels = [], []

    # Iterate through each category folder
    for folder in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder)
        if not folder.isdigit():
            print(f"Skipping invalid folder: {folder}")
            continue

        label = int(folder)
        for file in os.listdir(folder_path):
            try:
                img_path = os.path.join(folder_path, file)
                img = cv2.imread(img_path)
                img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
                img = img / 255.0  # Normalize pixel values
                images.append(img)
                labels.append(label)
            except Exception as e:
                print(f"Error processing file {file}: {e}")

    print(f"Loaded {len(images)} images with {len(set(labels))} unique labels.")
    return images, labels


def get_model():
    """
    Define and compile a convolutional neural network.
    """
    model = tf.keras.models.Sequential([
        # First convolutional layer
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu",
                               input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        # Second convolutional layer
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        # Flatten layers
        tf.keras.layers.Flatten(),

        # Fully connected layer
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.5),

        # Output layer
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    ])

    # Compile the model
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def get_callbacks():
    """
    Define callbacks for model training.
    """
    early_stopping = EarlyStopping(
        monitor='val_loss', patience=3, restore_best_weights=True
    )
    model_checkpoint = ModelCheckpoint(
        filepath="best_model.h5", save_best_only=True, monitor="val_loss"
    )
    return [early_stopping, model_checkpoint]


if __name__ == "__main__":
    main()
