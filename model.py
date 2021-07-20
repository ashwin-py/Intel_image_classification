from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, GlobalAveragePooling2D, Dropout
from tensorflow.keras import Sequential, Model, Input
from tensorflow.keras.layers.experimental.preprocessing import RandomRotation, RandomFlip, Rescaling
import tensorflow as tf


class CnnModel(Model):
    def __init__(self, num_of_classes, img_height, img_width):
        super().__init__(self)
        self.num_of_classes = num_of_classes
        self.img_height = img_height
        self.img_width = img_width

        self.data_augmentation = Sequential([
            RandomFlip('horizontal'),
            RandomRotation(0.2)
        ])

        self.resacle_layer = Rescaling(
            1. / 255,
            input_shape=(self.img_height, self.img_width, 3)
        )

    def load_model(self):
        model = Sequential([
            self.data_augmentation,
            self.resacle_layer,
            Conv2D(32, (3, 3), activation='relu'),
            MaxPool2D(),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPool2D(),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPool2D(),
            Flatten(),
            Dense(128, activation='relu'),
            Dense(self.num_of_classes, activation='softmax')
        ])
        return model

    def load_pretrained_model(self):

        base_model = tf.keras.applications.EfficientNetB7(
            include_top=False,
            input_shape=(150, 150, 3)
        )

        preprocess_input = tf.keras.applications.efficientnet.preprocess_input
        base_model.trainable = False

        inputs = Input(shape=(self.img_height, self.img_width, 3))
        x = self.data_augmentation(inputs)
        x = preprocess_input(x)
        x = base_model(x)
        x = GlobalAveragePooling2D()(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(64, activation='relu')(x)
        x = Dense(self.num_of_classes, activation='softmax')(x)

        model = Model(inputs=inputs, outputs=x)

        return model
