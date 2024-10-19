import pandas as pd
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

def load_image(file_path, target_size=(224, 224)):
    print(f"Attempting to load image: {file_path}")
    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"Warning: {file_path} does not exist. Skipping.")
        return None

    # Read the image
    image = cv2.imread(file_path)
    
    # Check if the image was loaded successfully
    if image is None:
        print(f"Warning: {file_path} could not be loaded (possibly corrupted). Skipping.")
        return None
    
    # Resize the image
    image_resized = cv2.resize(image, target_size)
    return image_resized

def load_data(csv_file, image_dir, target_size=(224, 224)):
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    X = []
    y = []
    
    for _, row in df.iterrows():
        # Construct the image file path
        image_path = os.path.join(image_dir, row['videoname'].replace('.mp4', '.jpg'))  # Replace mp4 with jpg
        label = 1 if row['label'] == 'FAKE' else 0  # 1 for FAKE, 0 for REAL
        
        # Load the image
        image = load_image(image_path, target_size)
        
        # Only append if the image is successfully loaded
        if image is not None:
            X.append(image)
            y.append(label)
    
    X = np.array(X)
    y = np.array(y)
    
    # Split the dataset into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_val, y_train, y_val