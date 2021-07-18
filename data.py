import os
import tensorflow as tf


class Data(object):
    def __init__(self, path):
        self.path = path
        self.classes = os.listdir(self.path)

    def show_count(self):
        count = [(class_, len(os.listdir(class_))) for class_ in self.classes]
        print(f'Number of instances : {count}')

    def split_train_test(self, validation_split):
        pass

    def load_dataset(self, validation_split, batch_size, image_size, cache=True):
        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            self.path,
            subset='training',
            validation_split=validation_split,
            batch_size=batch_size,
            image_size=image_size,
            seed=123
        )
        val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            self.path,
            subset='validation',
            validation_split=validation_split,
            batch_size=batch_size,
            image_size=image_size,
            seed=123
        )

        AUTOTUNE = tf.data.AUTOTUNE

        if cache:
            return train_ds.prefetch(buffer_size=AUTOTUNE), val_ds.prefetch(buffer_size=AUTOTUNE)
        else:
            return train_ds, val_ds
