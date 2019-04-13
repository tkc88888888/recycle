import os

from flask import Flask, request, redirect, url_for, jsonify, abort, Response
from flask import render_template
from werkzeug.utils import secure_filename

import skimage

import torch
import torchvision

import sqlite3

PRODUCT_NAME = "i-Bin"
PRODUCT_NAME_SHORT = "i-Bin"
UNLABELED_FOLDER = 'static/uploads/unlabeled/'
LABELED_FOLDER = 'static/uploads/labeled/'

IMAGE_LABELS = ["aluminium tin", "cardboard", "glass", "paper", "plastic"]

class_2_idx = {'aluminium tin': 0, 'cardboard': 1,
               'glass': 2, 'paper': 3, 'plastic': 4}
idx_2_class = {v: k for k, v in class_2_idx.items()}
print(idx_2_class)
model = torch.load("modeling/model.pth", map_location='cpu')
model = model.eval()
transforms = torchvision.transforms.Compose([
    torchvision.transforms.transforms.ToPILImage(),
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor()
])

os.makedirs(UNLABELED_FOLDER, exist_ok=True)
os.makedirs(LABELED_FOLDER, exist_ok=True)
for l in IMAGE_LABELS:
    os.makedirs(os.path.join(LABELED_FOLDER, l), exist_ok=True)

if not os.path.isfile('test.sqlite'):
    with sqlite3.connect('test.sqlite') as conn:
        conn.execute('''CREATE TABLE training_data
                (ID            INT             PRIMARY KEY,
                PATH           VARCHAR(255)    NOT NULL,
                LABEL          VARCHAR(50)     NOT NULL,
                WEIGHT         FLOAT);''')

    print("Created database and table")

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UNLABELED_FOLDER


@app.route('/')
def index():
    return render_template("base.html", **{
        "product_name": PRODUCT_NAME,
        "product_name_short": PRODUCT_NAME_SHORT
    })


@app.route('/label')
def label():
    return render_template("label.html", **{
        "product_name": PRODUCT_NAME,
        "product_name_short": PRODUCT_NAME_SHORT
    })


@app.route('/upload_training_image', methods=['GET', 'POST'])
def upload_training_image():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            return redirect(request.url)

        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return redirect(request.url)

    return render_template('upload.html', **{
        "product_name": PRODUCT_NAME,
        "product_name_short": PRODUCT_NAME_SHORT
    })


@app.route("/api/images/unlabeled/")
def get_unlabeled_images():
    """Get the list of unlabeled images."""
    unlabeled_images = [f for f in os.listdir(UNLABELED_FOLDER)]
    return jsonify(unlabeled_images)


@app.route("/api/images/set_label", methods=['POST'])
def set_image_label():
    if not request.json:
        return jsonify({
            "status": "Payload is not json."
        }), 400

    req_data = request.get_json()
    image = req_data["image"]
    print("image:", image)
    label = req_data["label"]
    print("label:", label)

    if label not in IMAGE_LABELS:
        return jsonify({
            "image": "Invalid label: " + label
        }), 400

    image_path = os.path.join(UNLABELED_FOLDER, image)

    if not os.path.isfile(image_path):
        return jsonify({
            "image": "File not found."
        }), 404

    target_path = os.path.join(LABELED_FOLDER, label, image)
    os.rename(image_path, target_path)

    with sqlite3.connect('test.sqlite') as conn:
        conn.execute("INSERT INTO training_data (PATH, LABEL, WEIGHT) \
            VALUES ('{}', '{}', {})".format(target_path, label, 1.0))
        conn.commit()
    # conn.close()

    if not image:
        return jsonify({
            "image": "This field is required."
        }), 400
    if not label:
        return jsonify({
            "label": "This field is required."
        }), 400

    return jsonify({
        "status": "Ok"
    })


@app.route("/api/images/predict", methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            return redirect(request.url)

        filename = secure_filename(file.filename)
        file.save(filename)

        img = skimage.io.imread(filename)

        img = transforms(img)
        print("img.shape:", img.shape)

        out = model(img.unsqueeze(0))
        print("out:", out)

        values, indices = out.max(1)
        print("values:", values)
        print("indices:", indices)
        print(idx_2_class[int(indices.item())])

        try:
            os.remove(filename)
        except:
            pass

        return jsonify({
            "prediction": idx_2_class[int(indices.item())]
        })

    return render_template('predict.html', **{
        "product_name": PRODUCT_NAME,
        "product_name_short": PRODUCT_NAME_SHORT,
    })
