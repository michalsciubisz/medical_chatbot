from PIL import Image
from tensorflow.keras.models import load_model

import numpy as np

def adjust_image(image):
    img = Image.open(image.stream)

    # converting to RGB
    if img.mode != 'RGB':
        img = img.convert('RGB')

    # adjusting size of the image to make prediction
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    return img_array

def predict_image_classification(image):
    model = load_model('D:/semestr_10/master_thesis/medical_chatbot/notebooks/models/radimagenet_best_model.h5')

    prediction = model.predict(image)

    return prediction
