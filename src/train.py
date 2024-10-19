import resource
resource.setrlimit(resource.RLIMIT_DATA, (10**9, 10**9))  # Limit set to 1GB
import logging
import sys
import os
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['TF_NUM_INTRAOP_THREADS'] = '2'
os.environ['TF_NUM_INTEROP_THREADS'] = '2'
from data_loader import load_data
from model import build_cnn
from tensorflow.keras.callbacks import ModelCheckpoint

# Set up logging
logging.basicConfig(level=logging.DEBUG)

print("Starting the training script...")
logging.debug("Loading dataset...")

def main():
    # Load the dataset
    X_train, X_val, y_train, y_val = load_data('data/metadata.csv', 'data/faces_224/')
    logging.debug("Dataset loaded successfully.")

    # Build the model
    model = build_cnn()
    logging.debug("Model built successfully.")

    # Define a checkpoint callback to save the best model
    checkpoint = ModelCheckpoint('models/cnn_model.h5', monitor='val_loss', save_best_only=True)

    # Train the model
    print("Starting model training...")
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=8, callbacks=[checkpoint])
    print("Model training completed!")

    # Save training logs
    logging.debug("Saving training logs...")
    with open('logs/training_log.txt', 'w') as f:
        f.write(str(history.history))
    logging.debug("Training logs saved.")

if __name__ == "__main__":
    main()