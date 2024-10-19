from src.data_loader import load_data
from tensorflow.keras.models import load_model

# Load the test data
_, X_test, _, y_test = load_data('data/dataset.csv', 'data/faces_224/')  # Ignore the training data, only load test data

# Load the trained model
try:
    model = load_model('models/cnn_model.h5')
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Evaluate the model
try:
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f'Test accuracy: {test_acc}')
except Exception as e:
    print(f"Error during model evaluation: {e}")