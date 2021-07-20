# Intel_image_classification
## Data
Data is used from kaggle link below,

https://www.kaggle.com/puneet6060/intel-image-classification/

## Model
Tried Different set of Models like:
1) ResNet152V2 
2) MobileNet
3) Vanilla CNN
4) EfficientNetB7

## Results
All Models fine-tuned for 50 epochs with Dropout Layer of 0.2 in Dense layer and 
had ~90% validation accuracy.
Using MobileNetV3 for deployment.
