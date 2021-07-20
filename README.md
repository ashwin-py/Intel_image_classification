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
All Models fine-tuned for 50 epochs with Dropout Layer of 0.2 in Dense layer(More details in model.py)
1. ResNet152 gave constant test accuracy of 90 throughout epoch end.
2. MobileNet also had 90-91 % accuracy throughout
## Deployment
Used Flask to deploy the model
## TODO
Use focal loss as loss function and test