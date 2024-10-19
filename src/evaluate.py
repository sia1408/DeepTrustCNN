from tensorflow.keras.models import load_model
from src.train import DataGenerator, load_metadata_and_split

def main():
    # Load metadata and split into test set (ignore training and validation data)
    _, X_test, _, y_test = load_metadata_and_split('data/metadata.csv', 'data/faces_224/')

    # Create the data generator for the test set
    test_generator = DataGenerator(X_test, y_test, batch_size=8, target_size=(224, 224), shuffle=False)

    # Load the trained model
    try:
        model = load_model('models/cnn_model.h5')
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        exit()

    # Evaluate the model using the test generator
    try:
        test_loss, test_acc = model.evaluate(test_generator)
        print(f'Test accuracy: {test_acc}')
    except Exception as e:
        print(f"Error during model evaluation: {e}")

if __name__ == "__main__":
    main()