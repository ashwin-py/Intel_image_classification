from flask import Flask, request, render_template
import os
from flask_cors import CORS, cross_origin
from classifier import Classifier

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

model_path = "./mobilenetv3"

app = Flask(__name__)
CORS(app)
classifier = Classifier(model=model_path)


@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')


@app.route("/submit", methods=['GET', 'POST'])
@cross_origin()
def predict():

    img = request.files['my_image']

    img_path = 'static/test.jpg'
    img.save(img_path)

    predictions = classifier.predict(img_path)
    return render_template("index.html", prediction=predictions, img_path=img_path)


if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5000, debug=True)