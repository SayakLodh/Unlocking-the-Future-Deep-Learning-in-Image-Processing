import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

def load_data(train_data_path, val_data_path):
    # Define image dimensions and color mode
    img_height = 224
    img_width = 224
    color_mode = 'grayscale'

    # Load training data
    train_datagen = ImageDataGenerator(rescale=1.0/255,
                                       rotation_range=15,
                                       width_shift_range=0.1,
                                       height_shift_range=0.1)

    train_generator = train_datagen.flow_from_directory(
        train_data_path,
        target_size=(img_height, img_width),
        color_mode=color_mode,
        batch_size=32,
        class_mode='binary')

    # Load validation data
    val_datagen = ImageDataGenerator(rescale=1.0/255)

    val_generator = val_datagen.flow_from_directory(
        val_data_path,
        target_size=(img_height, img_width),
        color_mode=color_mode,
        batch_size=32,
        class_mode='binary')

    return train_generator, val_generator

def create_model():
    # Create a deep learning model
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    return model

def train_model(model, train_generator, val_generator):
    # Compile and train the model
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(train_generator,
                        epochs=10,
                        validation_data=val_generator)
    return history

def evaluate_model(model, test_data_path):
    # Load test data
    test_datagen = ImageDataGenerator(rescale=1.0/255)
    test_generator = test_datagen.flow_from_directory(
        test_data_path,
        target_size=(224, 224),
        color_mode='grayscale',
        batch_size=32,
        class_mode='binary')

    # Evaluate the model on the test data
    test_loss, test_accuracy = model.evaluate(test_generator)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

def main():
    train_data_path = input("Enter path to training dataset: ").strip()
    val_data_path = input("Enter path to validation dataset: ").strip()
    test_data_path = input("Enter path to test dataset: ").strip()

    # Load training and validation data
    train_generator, val_generator = load_data(train_data_path, val_data_path)

    # Create and train the model
    model = create_model()
    train_model(model, train_generator, val_generator)
    print("Model trained successfully.")

    # Evaluate the model on the test dataset
    evaluate_model(model, test_data_path)

    model.save('pneumonia_model.h5')

if __name__ == "__main__":
    main()
