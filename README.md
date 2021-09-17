# Intel_image_classification
## Data
Data is used from kaggle link below,
https://www.kaggle.com/puneet6060/intel-image-classification/  

This Data contains around 25k images of size 150x150 distributed under 6 categories.  
{'buildings' -> 0,  
'forest' -> 1,  
'glacier' -> 2,  
'mountain' -> 3,  
'sea' -> 4,  
'street' -> 5 }  

The Train, Test and Prediction data is separated in each zip files. There are around 14k images in Train, 3k in Test and 7k in Prediction.  
This data was initially published on https://datahack.analyticsvidhya.com by Intel to host a Image classification Challenge.  

## Model
Experimented with Different set of Models like:
1) ResNet152V2 
2) MobileNet
3) Vanilla CNN
4) EfficientNetB7

## Results
All Models fine-tuned for 50 epochs with Dropout Layer of 0.2 in Dense layer and 
had ~90% validation accuracy.

## Deployment
Deployed using Flask and Using MobileNetV3 model.
