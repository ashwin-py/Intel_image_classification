import os
import tensorflow as tf


class Data(object):
    def __init__(self, train_path, val_path):
        self.train_path = train_path
        self.val_path = val_path
        self.classes = os.listdir(self.train_path)

    def show_count(self):
        count = [(class_, len(os.listdir(class_))) for class_ in self.classes]
        print(f'Number of instances : {count}')

    def load_dataset(self, validation_split, batch_size, image_size, cache=True):

        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            self.train_path,
            batch_size=batch_size,
            image_size=image_size,
            seed=123
        )
        val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            self.val_path,
            batch_size=batch_size,
            image_size=image_size,
            seed=123
        )

        AUTOTUNE = tf.data.AUTOTUNE

        if cache:
            return train_ds.prefetch(buffer_size=AUTOTUNE), val_ds.prefetch(buffer_size=AUTOTUNE)
        else:
            return train_ds, val_ds
