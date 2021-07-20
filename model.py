from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers.experimental.preprocessing import RandomRotation, RandomFlip, Rescaling
import tensorflow as tf


class CnnModel(Model):
    def __init__(self, num_of_classes, img_height, img_width):
        super().__init__(self)
        self.num_of_classes = num_of_classes
        self.img_height = img_height
        self.img_width = img_width

    def load_model(self):
        data_augmentation = Sequential([
            RandomFlip('horizontal'),
            RandomRotation(0.2)
        ])

        resacle_layer = Rescaling(1. / 255, input_shape=(self.img_height, self.img_width, 3))

        model = Sequential([
            data_augmentation,
            resacle_layer,
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

    def load_pretrained_model(self, model_name):
        base_model = eval(f"""tensorflow.keras.applications.{model_name}(
            include_top=False,
            input_shape=(150, 150, 3)
        )""")
        base_model.trainable = False

        x = GlobalAveragePooling2D()(base_model.output)
        x = Dense(512, activation='relu')(x)
        x = Dense(128, activation='relu')(x)
        x = Dense(self.num_of_classes, activation='softmax')(x)

        model = Model(inputs=base_model.input, outputs=x)

        return model
