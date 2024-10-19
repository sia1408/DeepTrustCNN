# Deepfake Detection using CNN (TensorFlow/Keras)

This project is a deepfake detection model built using a Convolutional Neural Network (CNN) implemented in TensorFlow/Keras.


## Setup

1. Clone the repository:
    ```
    git clone <repository_url>
    cd project-folder
    ```

2. Install the required packages:
    ```
    pip install -r requirements.txt
    ```

3. Train the model:
    ```
    python train.py
    ```

4. Check the saved model in the `models/` folder and training logs in the `logs/` folder.

## Customization

- You can modify the CNN architecture in `src/model.py` to improve performance.
- You can adjust the training parameters (e.g., epochs, batch size) in the `train.py` script or move them to a `config.py` file for flexibility.

