from packagename.registry import load_my_model
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array, load_img
#from tkinter import *
from PIL import Image
import numpy as np
from pathlib import Path

def preprocess(image):
    x=img_to_array(image)
    x_scaled=x/255
    x_scaled.resize((120, 120,3))
    reshaped_image = np.expand_dims(x_scaled, axis=0)
    return reshaped_image

def my_predict(image):
    model = load_my_model()

    prediction = model.predict(image)

    print(f'Probabilities of new cracks {prediction[0][0]}')

    return prediction[0][0]


if __name__ == "__main__":
    my_path = Path(__file__).parent.parent/"models/00001.jpg"
    image = load_img(my_path)
    reshaped_image = preprocess(image)

    my_predict(reshaped_image)
