import tensorflow as tf
import numpy as np


class Classifier:
    def __init__(self, model):
        self.model = tf.keras.models.load_model(model)
        self.image_size = 150
        self.classes = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

    def predict(self, image):
        img = tf.keras.preprocessing.image.load_img(
            image,
            target_size=(self.image_size, self.image_size)
        )
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)

        predictions = self.model.predict(img_array)
        predicted_class = self.classes[np.argmax(predictions)]
        return predictions, predicted_class
