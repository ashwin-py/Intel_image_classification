import streamlit as st
import pandas as pd
from classifier import Classifier

# header
st.title("Image Classification")

def predict(img_path):
    """
        A function to predict the class of the image
        Arg:
            img_path: str : takes a image path
        Returns:
            predictions: pandas dataframe
    """
    model_path = "./mobilenetv3" 
    classifier = Classifier(model_path)
    predictions = classifier.predict(img_path)

    return predictions


default_predicts_for_test_image = pd.DataFrame(
{'Classes': ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street'],
'Prediction Scores':['0.000%', '0.000%', '99.925%', '0.021%', '0.054%','0.000%']}
)


# two columns, col1 for image and col2 for result dataframe
col1, col2 = st.columns(2)

# widget for uploading a image file
img_path = st.file_uploader(
"Upload an image to test", 
type=["png", "jpg", "jpeg"]
)

if img_path is not None:
    prediction = predict(img_path)
    col1.image(img_path, caption=f"Actual Image", width=275)
    col2.dataframe(prediction)
else:
    col1.image('test.jpg', caption=f"Actual Image", width=275)
    col2.dataframe(default_predicts_for_test_image)
