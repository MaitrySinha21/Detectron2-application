"""
code : swati sinha & maitry sinha
"""
from flask import Flask, render_template, request
from flask_cors import cross_origin
from detectron import *
from utils import Decode
detector = Detector()
"""
read any bgr image with cv2 otherwise convert rgb-to-bgr
"""
app = Flask(__name__)


@app.route('/', methods=['GET'])
@cross_origin()
def index():
    return render_template('index.html')


@app.route('/result', methods=['GET', 'POST'])
@cross_origin()
def result():
    if request.method == 'POST':
        image = request.json['image']
        img = Decode(image).copy()
        detector.segmentation(img)
        detector.detection(img)
        return render_template('index.html')
    return render_template('index.html')


if __name__ == "__main__":
    app.run()
