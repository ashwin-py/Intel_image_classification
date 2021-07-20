from data import Data
from model import CnnModel
from trainer import Trainer

BATCH_SIZE = 64
IMG_SIZE = 150
EPOCHS = 50


data_obj = Data('./data/seg_train')
classes = data_obj.classes

train_ds, val_ds = data_obj.load_dataset(0.2, BATCH_SIZE, (IMG_SIZE, IMG_SIZE))

model_obj = CnnModel(len(classes), IMG_SIZE, IMG_SIZE)
"""
Model Names:
Xception	, InceptionV3
VGG16	    , InceptionResNetV2
VGG19	    , MobileNet
ResNet50	, MobileNetV2
ResNet101	, DenseNet121
ResNet152	, DenseNet169
ResNet50V2	, DenseNet201
ResNet101V2	, NASNetMobile
ResNet152V2	, NASNetLarge

"""

model = model_obj.load_pretrained_model("ResNet152V2")
# print(type(model))
trainer = Trainer(model, train_ds, val_ds, BATCH_SIZE, EPOCHS)

callback1 = trainer.tensorboard_callback(log_dir='./logs')
callback2 = trainer.save_checkpoint(path='./checkpoints')
# for img, label in train_ds.take(1):
#     print(img, label)
trainer.train(callbacks=[callback1, callback2])
