import pandas as pd
import sys
import cv2
import numpy as np
from tensorflow.keras.utils import Sequence
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.callbacks import ModelCheckpoint
from src.model import build_cnn
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Custom data generator class
class DataGenerator(Sequence):
    def __init__(self, image_paths, labels, batch_size=32, target_size=(224, 224), shuffle=True):
        self.image_paths = image_paths
        self.labels = labels
        self.batch_size = batch_size
        self.target_size = target_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        # Returns the number of batches per epoch
        return int(np.floor(len(self.image_paths) / self.batch_size))

    def __getitem__(self, index):
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of image_paths and labels
        batch_image_paths = [self.image_paths[k] for k in indexes]
        batch_labels = [self.labels[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(batch_image_paths, batch_labels)

        return X, y

    def on_epoch_end(self):
        # Updates indexes after each epoch
        self.indexes = np.arange(len(self.image_paths))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, batch_image_paths, batch_labels):
        # Generates data containing batch_size samples
        X = np.empty((self.batch_size, *self.target_size, 3))
        y = np.empty((self.batch_size), dtype=int)

        for i, (img_path, label) in enumerate(zip(batch_image_paths, batch_labels)):
            # Load image
            image = cv2.imread(img_path)
            if image is not None:
                image = cv2.resize(image, self.target_size)
                image = img_to_array(image) / 255.0  # Normalize
                X[i,] = image
            else:
                print(f"Warning: Could not load image {img_path}. Skipping.")

            # Label
            y[i] = label

        return X, y

def load_metadata_and_split(csv_file, image_dir):
    df = pd.read_csv(csv_file)
    
    image_paths = []
    labels = []

    for _, row in df.iterrows():
        # Create the image path
        image_path = os.path.join(image_dir, row['videoname'].replace('.mp4', '.jpg'))
        image_paths.append(image_path)

        # Set the label (1 for FAKE, 0 for REAL)
        label = 1 if row['label'] == 'FAKE' else 0
        labels.append(label)

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(image_paths, labels, test_size=0.2, random_state=42)

    return X_train, X_val, y_train, y_val

def main():
    # Load metadata and split into train/val sets
    X_train, X_val, y_train, y_val = load_metadata_and_split('data/metadata.csv', 'data/faces_224/')

    # Create data generators
    train_generator = DataGenerator(X_train, y_train, batch_size=8, target_size=(224, 224))
    val_generator = DataGenerator(X_val, y_val, batch_size=8, target_size=(224, 224))

    # Build the CNN model
    model = build_cnn()

    # Define checkpoint callback to save the best model
    checkpoint = ModelCheckpoint('models/cnn_model.h5', monitor='val_loss', save_best_only=True)

    # Train the model using the data generators
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=10,
        callbacks=[checkpoint]
    )

    print("Model training completed!")

    # Save training logs if needed
    with open('logs/training_log.txt', 'w') as f:
        f.write(str(history.history))

if __name__ == "__main__":
    main()
