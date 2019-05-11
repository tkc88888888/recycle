import csv
import datetime
import errno
import io
import json
import os
import time

import MySQLdb
import cv2
# import request
# import parse
import flask
import keras
import imutils
import pandas as pd
from PIL import Image
from flask import Flask
from imutils import contours
from imutils import perspective
from keras.models import load_model
from scipy.spatial import distance as dist
from skimage import io
from skimage.measure import compare_ssim

try:
    from urllib.parse import urlparse
except ImportError:
    from urlparse import urlparse
# import urllib.parse


try:  # python3
    from urllib.request import urlopen
except:  # python2
    from urllib2 import urlopen
# import urllib.request

# import urllib.response

import numpy as np
import tensorflow as tf

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024

global datestring

datestring = time.strftime("%Y_%m_%d_%H:%M")
attrpath = "/home/kc/Downloads/keras-multi-input/new_imagepath_and_attributes_come_in_here_row_by_row.csv"
modelpath = "/home/kc/Downloads/keras-multi-input/model.hdf5"
labelpath = "/home/kc/Downloads/keras-multi-input/kerasmodel.txt"
tmppath = "tmp"

image0top = "/home/kc/Downloads/keras-multi-input/latest-without-waste-top.jpg"
image1top = "/home/kc/Downloads/keras-multi-input/Collected_Top_Images/image1top.jpg"
image0front = "/home/kc/Downloads/keras-multi-input/latest-without-waste-front.jpg"
image1front = "/home/kc/Downloads/keras-multi-input/Collected_Front_Images/image1front.jpg"


####################################################################################
# FROM SIZER.PY
####################################################################################

def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)


def preprocess_image(image0, image1, X, Y):
    # load the two input images
    loaded0 = cv2.imread(image0)
    loaded1 = cv2.imread(image1)

    # convert the images to grayscale
    gray0 = cv2.cvtColor(loaded0, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.cvtColor(loaded1, cv2.COLOR_BGR2GRAY)

    # compute the Structural Similarity Index (SSIM) between the two
    # images, ensuring that the difference image is returned
    (scoreA, diffA) = compare_ssim(gray0, gray1, full=True)
    img = (diffA * 255).astype("uint8")
    img = cv2.resize(img, (X, Y))

    image1 = cv2.GaussianBlur(img, (11, 11), 0)
    image2 = cv2.dilate(image1, None, iterations=10)
    image3 = cv2.erode(image2, None, iterations=10)

    (scoreB, diffB) = compare_ssim(image1, image3, full=True)
    image = (diffB * 255).astype("uint8")

    # threshold the difference image, followed by finding contours to
    # obtain the regions of the two input images that differ
    thresh = cv2.threshold(image, 0, 255,
                           cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # sort the contours from left-to-right and initialize the
    # 'pixels per metric' calibration variable
    (cnts, _) = contours.sort_contours(cnts)

    return cnts, image

def findmaxdimension(cnts, image, pixelsPerMetric):
    dimensions = []
    # loop over the contours individually
    for c in cnts:
        # if the contour is not sufficiently large, ignore it
        if cv2.contourArea(c) > 1000000:
            continue

        # compute the rotated bounding box of the contour
        orig = image.copy()
        box = cv2.minAreaRect(c)
        box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
        box = np.array(box, dtype="int")

        # order the points in the contour such that they appear
        # in top-left, top-right, bottom-right, and bottom-left
        # order, then draw the outline of the rotated bounding
        # box
        box = perspective.order_points(box)
        cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)

        # loop over the original points and draw them
        for (x, y) in box:
            cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)

        # unpack the ordered bounding box, then compute the midpoint
        # between the top-left and top-right coordinates, followed by
        # the midpoint between bottom-left and bottom-right coordinates
        (tl, tr, br, bl) = box
        (tltrX, tltrY) = midpoint(tl, tr)
        (blbrX, blbrY) = midpoint(bl, br)

        # compute the midpoint between the top-left and top-right points,
        # followed by the midpoint between the top-righ and bottom-right
        (tlblX, tlblY) = midpoint(tl, bl)
        (trbrX, trbrY) = midpoint(tr, br)

        # draw the midpoints on the image
        cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
        cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
        cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
        cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

        # draw lines between the midpoints
        cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
                 (255, 0, 255), 2)
        cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
                 (255, 0, 255), 2)

        # compute the Euclidean distance between the midpoints
        dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))  # vertical width
        dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))  # horizontal length

        # compute the size of the object
        dimA = dA / pixelsPerMetric
        dimB = dB / pixelsPerMetric

        dim = dimA * dimB

        if image.shape[0] == 288:
            if 0 < blbrX < 352:
                if 0 < blbrY < 288:
                    dimensions.append(dim)

        elif image.shape[0] == 648:
            if 0 < blbrX < 864:
                if tlblX < 432:
                    if 432 < trbrX:
                        if 360 < blbrY < 415:
                            dimensions.append(dim)

    maxdim = max(dimensions)

    if image.shape[0] == 288:
        print ("Top Area is : {}cm2".format(maxdim))

    elif image.shape[0] == 648:
        print ("Front Area is : {}cm2".format(maxdim))

    return maxdim


def drawline(cnts, image, pixelsPerMetric, maxdim, saveimg):
    for c in cnts:
        # if the contour is not sufficiently large, ignore it
        if cv2.contourArea(c) > 1000000:
            continue

        # compute the rotated bounding box of the contour
        orig = image.copy()
        box = cv2.minAreaRect(c)
        box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
        box = np.array(box, dtype="int")

        # order the points in the contour such that they appear
        # in top-left, top-right, bottom-right, and bottom-left
        # order, then draw the outline of the rotated bounding
        # box
        box = perspective.order_points(box)
        cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)

        # loop over the original points and draw them
        for (x, y) in box:
            cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)

        # unpack the ordered bounding box, then compute the midpoint
        # between the top-left and top-right coordinates, followed by
        # the midpoint between bottom-left and bottom-right coordinates
        (tl, tr, br, bl) = box
        (tltrX, tltrY) = midpoint(tl, tr)
        (blbrX, blbrY) = midpoint(bl, br)

        # compute the midpoint between the top-left and top-right points,
        # followed by the midpoint between the top-righ and bottom-right
        (tlblX, tlblY) = midpoint(tl, bl)
        (trbrX, trbrY) = midpoint(tr, br)

        # draw the midpoints on the image
        cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
        cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
        cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
        cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

        # draw lines between the midpoints
        cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
                 (255, 0, 255), 2)
        cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
                 (255, 0, 255), 2)

        # compute the Euclidean distance between the midpoints
        dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))  # vertical width
        dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))  # horizontal length

        # if the pixels per metric has not been initialized, then
        # compute it as the ratio of pixels to supplied metric
        # (in this case, cm)
        if pixelsPerMetric is None:
            pixelsPerMetric = 240 / 14.4  # 236 pixel, 14.4 is actual

        # compute the size of the object
        dimA = dA / pixelsPerMetric
        dimB = dB / pixelsPerMetric

        dim = dimA * dimB

        # draw the object sizes on the image
        cv2.putText(orig, "{:.1f}cm".format(dimB),
                    (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 0), 2)
        cv2.putText(orig, "{:.1f}cm".format(dimA),
                    (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 0), 2)

        # show the output image
        cv2.imshow("Image", orig)

        cv2.imwrite(top, orig)

        brightness = np.mean(image)
        # print brightness

        if dim == maxdim:
            if image.shape[0] == 288:

                print ("Vision length is : {}cm".format(image.shape[1] / pixelsPerMetric))
                print ("Vision width is : {}cm".format(image.shape[0] / pixelsPerMetric))

                if brightness > 140:
                    return (dimB, dimA)

                else:
                    print ("Huge item seen from top, estimated whole vision area")
                    dimA = image.shape[0] / pixelsPerMetric
                    dimB = image.shape[1] / pixelsPerMetric
                    return (dimB, dimA)

            if image.shape[0] == 648:

                print ("Vision height is : {}cm".format(image.shape[0] / pixelsPerMetric))

                if brightness > 180:
                    return dimA

                else:
                    print ("Huge item seen from front, estimated whole vision area")
                    dimA = image.shape[0] / pixelsPerMetric
                    return dimA


def getsize(image0top, image1top, image0front, image1front):
    topX = 352
    topY = 288
    frontX = 864
    frontY = 648
    pixelsPerMetricTop = 16.67
    pixelsPerMetricFront = 28.84

    try:
        (cntsTop, imageTop) = preprocess_image(image0top, image1top, topX, topY)
        maxdimTop = findmaxdimension(cntsTop, imageTop, pixelsPerMetricTop)
        (length, width) = drawline(cntsTop, imageTop, pixelsPerMetricTop, maxdimTop)
        print ("Length is : {}".format(length))
        print ("Width is : {}".format(width))

    except:
        length = 0.01
        width = 0.01
        print ("Top Waste Area too Small")

    try:
        (cntsFront, imageFront) = preprocess_image(image0front, image1front, frontX, frontY)
        maxdimFront = findmaxdimension(cntsFront, imageFront, pixelsPerMetricFront)
        height = drawline(cntsFront, imageFront, pixelsPerMetricFront, maxdimFront)
        print ("Height is : {}".format(height))

    except:
        height = 0.01
        print ("Front Waste Area too Small")

    volume = length * width * height
    Volume = int(volume)
    print ("Volume is : {}".format(Volume))
    return (length, width, height)


####################################################################################
# FROM SERVING.PY
####################################################################################

def load_graph(model_file):
    graph = tf.Graph()
    graph_def = tf.GraphDef()

    with open(model_file, "rb") as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)
    return graph


@app.route('/classify_image', methods=['POST'])
def classify_image():
    files = flask.request.files
    print(files)
    print(len(files))
    if flask.request.method == "POST":
        if flask.request.files.get("image1top"):
            # read the image in PIL format
            image = flask.request.files.get("image1top").read()
            image = Image.open(io.BytesIO(image))
            print('saved image1top')
            image.save(image1top, quality=80, optimize=True, progressive=True)

    if flask.request.method == "POST":
        if flask.request.files.get("image1front"):
            image = flask.request.files.get("image1front").read()
            image = Image.open(io.BytesIO(image))
            print('saved image1front')
            image.save(image1front, quality=80, optimize=True, progressive=True)

    if flask.request.method == "POST":
        if flask.request.files.get("image0top"):
            # read the image in PIL format
            image = flask.request.files.get("image0top").read()
            image = Image.open(io.BytesIO(image))
            print('saved image0top')
            image.save(image0top, quality=80, optimize=True, progressive=True)

    if flask.request.method == "POST":
        if flask.request.files.get("image0front"):
            image = flask.request.files.get("image0front").read()
            image = Image.open(io.BytesIO(image))
            print('saved image0front')
            image.save(image0front, quality=80, optimize=True, progressive=True)

    (length, width, height) = getsize(image0top, image1top, image0front, image1front)

    if flask.request.method == "POST":
        if flask.request.form.get("weightattr"):
            attrpth = saveattr(attrpath, float(flask.request.form.get("weightattr")), length, width, height)

    df = attrpreprocess(attrpth)
    image = imagepreprocess(image1top)
    # model loaded externally faster
    labels = loadlabels(labelpath)
    categoryscore = predict(df, image, model)
    result = {"Category": categoryscore[0], "Probability": categoryscore[1]}

    print (result)
    return json.dumps(result)


@app.route('/status', methods=['GET'])
def status():
    return 'online'


@app.route('/', methods=['GET'])
def index():
    return status()


def create_tmp(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


def saveattr(attrpath, weightattr, length, width, height):
    global datestring
    attrdata = []

    attrdata.append(weightattr)
    print ("Weight attributes data received are : {}".format(weightattr))

    attrdata.append(length)
    print ("Length attribute data received are : {}".format(length))

    attrdata.append(width)
    print ("Width attribute data received are : {}".format(width))

    attrdata.append(height)
    print ("Height attribute data received are : {}".format(height))

    print attrdata

    with open(attrpath, 'w') as csvFile:
        writer = csv.writer(csvFile, dialect='excel')
        writer.writerow(attrdata)

    csvFile.close()

    print ("Attributes data written to : {}".format(attrpath))
    return attrpath


def attrpreprocess(attrpath):
    cols = ["weight", "top_area", "front_area", "time"]
    df = pd.read_csv(attrpath, sep=",", header=None, names=cols)
    print df

    minweight = 0
    maxweight = 300
    rangeweight = maxweight - minweight

    mintoparea = 0
    maxtoparea = 600
    rangetoparea = maxtoparea - mintoparea

    minfrontarea = 0
    maxfrontarea = 600
    rangefrontarea = maxfrontarea - minfrontarea

    mintime = 7  # 7am
    maxtime = 18  # 6pm
    rangetime = maxtime - mintime

    df["weight"] = df["weight"] - minweight
    df["weight"] = df["weight"] / rangeweight
    print df

    df["top_area"] = df["top_area"] - mintoparea
    df["top_area"] = df["top_area"] / rangetoparea

    df["front_area"] = df["front_area"] - minfrontarea
    df["front_area"] = df["front_area"] / rangefrontarea

    df["time"] = df["time"] - mintime
    df["time"] = df["time"] / rangetime
    print ("this is df")
    print df
    return df


def imagepreprocess(imagepath):
    image = []
    img = cv2.imread(imagepath)
    img = cv2.resize(img, (224, 224))
    image.append(img)
    image = np.array(image, dtype="float") / 255.0
    print ("this is image")
    print image
    return image


def loadmodel(modelpath):
    print("[INFO] loading model...")
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.99999
    sess = tf.Session(config=config)
    keras.backend.set_session(sess)
    model = load_model(modelpath)
    return model


def loadlabels(labelpath):
    labels = []
    proto_as_ascii_lines = tf.gfile.GFile(labelpath).readlines()
    for l in proto_as_ascii_lines:
        labels.append(l.rstrip())
    print ("this is labels")
    print labels
    return labels


def predict(df, image, model):
    print("[INFO] predicting waste category...")
    prediction = model.predict([df, image])
    print ("this is prediction before squeeze")
    print prediction
    preds = np.squeeze(prediction)
    print ("this is preds after squeeze")
    print preds
    arrangedindex = preds.argsort()[-1:][::-1]
    labels = loadlabels(labelpath)
    categoryscore = []
    for i in arrangedindex:
        # print(labels[i], results[i])
        category = labels[i]
        score = preds[i]
        categoryscore.append('%s' % (category))
        categoryscore.append('%.5f' % (score * 100))
    print ("this is categoryscore")
    print categoryscore
    return categoryscore


def saveDB(imgpath, weight, length, width, height):
    t = datetime.datetime.now().strftime(
        '%Y-%m-%d %H:%M:%S')  # get time infomation in year month date hour minute seconds format
    sql = "INSERT INTO data_log (Imgpath, Date_Time,weight,length,width,height) VALUES (%s,%s,%s,%s,%s,%s)"  # to simplify the info insertion command(looks more neat), can actually write in c.execute line
    val = (imgpath, t, weight, length, width, height)  # arguments for input info, reading each variable as string.

    try:  # try to execute insertion of info to databse and commit/save it, except when it fails, rollback/restore previous database state
        c.execute(sql, val)
        db.commit()

    except:
        db.rollback()


create_tmp('tmp')
model = loadmodel(modelpath)
model._make_predict_function()

try:
    db = MySQLdb.connect("localhost", "remote", "remote",
                         "mydb")  # Calling imported module mysqldb to allow connection, input arguments are self set: host= localhost, username=remote, password=remote, database=mydb
    c = db.cursor()
except:
    print('database connection error')

app.run(threaded=True, host='0.0.0.0', port=os.environ.get('PORT', 8000))
