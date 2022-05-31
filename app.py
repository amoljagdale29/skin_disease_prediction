import os
import uuid
import flask
import urllib
from flask import Flask, render_template, request,redirect
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf

import glob
import cv2
from tensorflow import keras
import numpy as np

skin_model = keras.models.load_model("skin_disease_model.h5", compile=True)

LABEL_MAP = ['Acne and Rosacea Photos',
             'Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions',
             'Eczema Photos',
             'Melanoma Skin Cancer Nevi and Moles',
             'Psoriasis pictures Lichen Planus and related diseases',
             'Tinea Ringworm Candidiasis and other Fungal Infections',
             'Urticaria Hives',
             'Nail Fungus and other Nail Disease']


def detect(filename):
    img = load_img(filename, target_size=(32, 32))
    img = img_to_array(img)

    image = cv2.resize(img, (32, 32)) / 255.0
    prediction = skin_model.predict(image.reshape(1, 32, 32, 3), verbose=0)
    max_score = prediction[0][np.argsort(prediction[0])][::-1][:3]*100
    max_class = np.argsort(prediction[0])[::-1][:3]
    max_score = np.around(max_score, decimals=2)
    out_class = []
    for ind in max_class:
        out_class.append(LABEL_MAP[ind])
    return out_class, max_score


app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ALLOWED_EXT = set(['jpg', 'jpeg', 'png', 'jfif'])


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXT


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/success', methods=['GET', 'POST'])
def success():
    error = ''
    target_img = os.path.join(os.getcwd(), 'static/images')
    if request.method == 'POST':
        if (request.form):
            link = request.form.get('link')
            try:
                resource = urllib.request.urlopen(link)
                unique_filename = str(uuid.uuid4())
                filename = unique_filename + ".jpg"
                img_path = os.path.join(target_img, filename)
                output = open(img_path, "wb")
                output.write(resource.read())
                output.close()
                img = filename

                class_result, prob_result = detect(img_path)

                predictions = {
                    "class1": class_result[0],
                    "class2": class_result[1],
                    "class3": class_result[2],
                    "prob1": prob_result[0],
                    "prob2": prob_result[1],
                    "prob3": prob_result[2],
                }

            except Exception as e:
                print(str(e))
                error = 'This image from this site is not accesible or inappropriate input'

            if (len(error) == 0):
                print(predictions)
                return render_template('success.html', img=img, predictions=predictions)
            else:
                return redirect('/', error=error)


        elif (request.files):
            file = request.files['file']
            if file and allowed_file(file.filename):
                file.save(os.path.join(target_img, file.filename))
                img_path = os.path.join(target_img, file.filename)
                img = file.filename

                class_result, prob_result = detect(img_path)

                predictions = {
                    "class1": class_result[0],
                    "class2": class_result[1],
                    "class3": class_result[2],
                    "prob1": prob_result[0],
                    "prob2": prob_result[1],
                    "prob3": prob_result[2],
                }

            else:
                error = "Please upload images of jpg , jpeg and png extension only"

            if (len(error) == 0):
                return render_template('success.html', img=img, predictions=predictions)
            else:
                return redirect('/', error=error)

    else:
        return redirect('/', error=error)


if __name__ == "__main__":
    app.run(debug=True)
