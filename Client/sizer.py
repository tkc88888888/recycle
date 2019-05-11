# import the necessary packages

import cv2
import imutils
import numpy as np
from imutils import contours
from imutils import perspective
from scipy.spatial import distance as dist
from skimage.measure import compare_ssim


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
    pixelsPerMetric = None

    return (cnts, image)


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
    global top
    global front

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

        # print dB
        # print dA
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
        cv2.imwrite(saveimg, orig)

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


def main(image0top, image1top, image0front, image1front, saveTop, saveFront):
    topX = 352
    topY = 288
    frontX = 864
    frontY = 648
    pixelsPerMetricTop = 16.67
    pixelsPerMetricFront = 28.84

    try:
        (cntsTop, imageTop) = preprocess_image(image0top, image1top, topX, topY)
        maxdimTop = findmaxdimension(cntsTop, imageTop, pixelsPerMetricTop)
        (length, width) = drawline(cntsTop, imageTop, pixelsPerMetricTop, maxdimTop, saveTop)
        print ("Length is : {}".format(length))
        print ("Width is : {}".format(width))

    except:
        length = 0.01
        width = 0.01
        print ("Top Waste Area too Small")

    try:
        (cntsFront, imageFront) = preprocess_image(image0front, image1front, frontX, frontY)
        maxdimFront = findmaxdimension(cntsFront, imageFront, pixelsPerMetricFront)
        height = drawline(cntsFront, imageFront, pixelsPerMetricFront, maxdimFront, saveFront)
        print ("Height is : {}".format(height))

    except:
        height = 0.01
        print ("Front Waste Area too Small")

    volume = length * width * height
    Volume = round(volume)
    print ("Volume is : {}cm3".format(Volume))
    return (length, width, height)


image0top = "/home/pi/4.jpg"
image1top = "/home/pi/2.jpg"
image0front = "/home/pi/3.jpg"
image1front = "/home/pi/1.jpg"
saveTop = "/home/pi/plasticTop.jpg"
saveFront = "/home/pi/plasticFront.jpg"
main(image0top, image1top, image0front, image1front, saveTop, saveFront)
