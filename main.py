from data import Data
from model import CnnModel
from trainer import Trainer
import tensorflow as tf

if tf.test.is_gpu_available():
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


BATCH_SIZE = 64
IMG_SIZE = 150
EPOCHS = 50
VALIDATION_SPLIT = 0.2

data_obj = Data('./dataset/seg_train', './dataset/seg_test')
classes = data_obj.classes

train_ds, val_ds = data_obj.load_dataset(VALIDATION_SPLIT, BATCH_SIZE, (IMG_SIZE, IMG_SIZE))

model_obj = CnnModel(len(classes), IMG_SIZE, IMG_SIZE)

model = model_obj.load_pretrained_model()  # Loads ResNet152V2 by default

trainer = Trainer(model, train_ds, val_ds, BATCH_SIZE, EPOCHS)

callback1 = trainer.tensorboard_logs(log_dir='./logs')
callback2 = trainer.save_checkpoint(path='./checkpoints')
# for img, label in train_ds.take(1):
#     print(img, label)
if __name__ == '__main__':

    trainer.train(callbacks=[callback1, callback2])
