from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the model
model = load_model('tb_model.h5')

# Define image preprocessing
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224), color_mode='grayscale')
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

# Define prediction function
def predict_tb(model, img_path):
    img_array = preprocess_image(img_path)
    prediction = model.predict(img_array)
    if prediction[0] > 0.5:
        print("Prediction: TB Detected")
    else:
        print("Prediction:Normal")

# Run prediction on a sample image
img_path = 'D:\\vs\\project7\\tb\\TB_Chest_Radiography_Database\\test\\tb\\Tuberculosis-541.png'
predict_tb(model, img_path)
