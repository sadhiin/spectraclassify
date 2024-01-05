"""
This file is responsible for the prediction service of the latest trained model define by the user.
"""

import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from spectraclassify.utility.data_manager import preprocess_input_image


class PrecictionService:
    def __init__(self, model_path: str, classes: dict):
        self.model = load_model(model_path)
        self.classes = classes

    def predict(self, image_path: str):
        try:
            x = preprocess_input_image(image_path)
            prediction = self.model.predict(x)
            prediction = np.argmax(prediction, axis=1)
            return self.classes[prediction[0]]
        except Exception as e:
            print(f"Error in prediction: {e}")
            return None
