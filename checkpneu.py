from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the model
model = load_model('pneumonia_model.h5')

# Define image preprocessing
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224), color_mode='grayscale')
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

# Define prediction function
def predict_pneumonia(model, img_path):
    img_array = preprocess_image(img_path)
    prediction = model.predict(img_array)
    if prediction[0] > 0.5:
        print("Prediction: Pneumonia Detected")
    else:
        print("Prediction: Normal")

# Run prediction on a sample image
img_path = 'D:\\vs\\project7\\chest_xray\\chest_xray\\test\\PNEUMONIA\\person1_virus_6.jpeg'
predict_pneumonia(model, img_path)
