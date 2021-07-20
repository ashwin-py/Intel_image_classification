from flask import Flask, request, jsonify, render_template
import os
from flask_cors import CORS, cross_origin
from utils import decode_image
from classifier import Classifier

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

model_path = "./models/test.h5"

app = Flask(__name__)
CORS(app)
classifier = Classifier(model=model_path)


@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')


@app.route("/predict", methods=['POST'])
@cross_origin()
def predict_route():
    if request.method == 'POST':
        image = request.files['file']
        image_path = "static/" + image.filename
        image.save('sample.jpg')
    predtictions, predicted_class = classifier.predict("sample.jpg")

    return jsonify({"text": predicted_class})


if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5000, debug=True)