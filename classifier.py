import tensorflow as tf
import pandas as pd

# if tf.test.is_gpu_available():
#     physical_devices = tf.config.list_physical_devices('GPU')
#     tf.config.experimental.set_memory_growth(physical_devices[0], True)


class Classifier:
    def __init__(self, model):
        self.model = tf.keras.models.load_model(model)
        self.image_size = 150
        self.classes = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

    def predict(self, image):
        """
        A fucntion to predict an image 
        Args: 
            image: str: Takes a image path 

        Returns predictions in pandas dataframe
        """
        img_array = tf.image.resize(image, (150, 150), method='nearest')
        img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
        img_array = tf.expand_dims(img_array, 0)

        predictions = self.model.predict(img_array)
        predictions = ["{:.3%}".format(score) for score in predictions[0]]
        df = pd.DataFrame(data={'Classes': self.classes, 'Prediction Scores': predictions})
        return df

