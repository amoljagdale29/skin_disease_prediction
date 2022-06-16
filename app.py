import os
import uuid
import urllib
from flask import Flask, render_template, request, redirect
import cv2
from detect import pre_process, post_process
import numpy as np

modelWeights = 'acne_classification_v1.onnx'
net = cv2.dnn.readNet(modelWeights)
classes = ['Level_0', 'Level_1', 'Level_2']


def detect(filename, resource):
    # read image from
    f = resource.read()
    npimg = np.fromstring(f, np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    # Process image.
    detections = pre_process(frame, net)
    img, result = post_process(frame.copy(), detections)
    # save image to load in front end
    cv2.imwrite(filename, img)
    return img


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
        if request.form:
            link = request.form.get('link')
            try:
                resource = urllib.request.urlopen(link)
                unique_filename = str(uuid.uuid4())
                filename = unique_filename + ".jpg"
                img_path = os.path.join(target_img, filename)
                img = detect(img_path, resource)
                img = filename
            except Exception as e:
                print(str(e))
                error = 'This image from this site is not accesible or inappropriate input'

            if len(error) == 0:
                return render_template('success.html', img=img)
            else:
                return redirect('/')

        elif request.files:
            file = request.files['file']
            if file and allowed_file(file.filename):
                img_path = os.path.join(target_img, file.filename)
                img = file.filename
                detect(img_path, file)
            else:
                error = "Please upload images of jpg , jpeg and png extension only"
            if len(error) == 0:
                return render_template('success.html', img=img)
            else:
                return redirect('/')
    else:
        return redirect('/')


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5020)
