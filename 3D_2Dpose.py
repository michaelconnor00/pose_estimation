"""
Usage:
    3D_2Dpose.py [<file_name>]

Script for estimating 3D coordinates of a marker from a 2D image
and the marker resource.

Arguments:
    <file_name>     The name of the image file

Options:
    --verbose       Print debug output

"""
import os
import numpy as np
import cv2
from docopt import docopt

def main(arguments):
    print arguments
    image_filename = arguments['<file_name>']

    cwd = os.getcwd()

    img = cv2.imread(os.path.join(cwd, image_filename))

    # http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_feature2d/py_shi_tomasi/py_shi_tomasi.html
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    corners = cv2.goodFeaturesToTrack(gray, 25, 0.01, 10)
    corners = np.int0(corners)

    for i in corners:
        x, y = i.ravel()
        cv2.circle(img, (x,y), 3, 255, -1)

    cv2.imwrite('orig_w_corners.jpg', img)

    # # http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_fast/py_fast.html
    # # Initiate FAST object with default values
    # fast = cv2.FastFeatureDetector()
    #
    # # find and draw the keypoints
    # kp = fast.detect(img,None)
    # img2 = cv2.drawKeypoints(img, kp, color=(255,0,0))
    #
    # print "Threshold: ", fast.getInt('threshold')
    # print "nonmaxSuppression: ", fast.getBool('nonmaxSuppression')
    # # print "neighborhood: ", fast.getInt('type')
    # print "Total Keypoints with nonmaxSuppression: ", len(kp)
    #
    # cv2.imwrite('fast_true.png',img2)

    # # http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_features_harris/py_features_harris.html
    # img = cv2.imread(image_filename)
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #
    # gray = np.float32(gray)
    #
    # # img - Input image, it should be grayscale and float32 type.
    # # blockSize - It is the size of neighbourhood considered for corner detection
    # # ksize - Aperture parameter of Sobel derivative used.
    # # k - Harris detector free parameter in the equation.
    # cv2.cornerHarris(gray, 2, 3, 0.04)
    #
    # # Result is dilated for marking the corners, not important
    # dst = cv2.dilate(img, None)
    #
    # # Threshold for an optimal value, it may vary depending on the image.
    # img[dst > 0.01 * dst.max()] = [0,0,255]

    cv2.imshow('image',img)
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()

    # # Draw a diagonal blue line with thickness of 3 px
    # # For marker32.png, corners of the marker are (160, 3*160) (160, 4*160) (2*160, 3*160) (2*160, 4*160)
    # # image is 800x800 with 5x5 grid of white and black blocks, a single white block at col=2, row=4
    # cv2.line(img, (160, 3*160), (160, 4*160), (255, 0, 0), 3)
    # cv2.line(img, (160, 3*160), (2*160, 3*160), (255, 0, 0), 3)
    # cv2.line(img, (2*160, 3*160), (2*160, 4*160), (255, 0, 0), 3)
    # cv2.line(img, (160, 4*160), (2*160, 4*160), (255, 0, 0), 3)


if __name__ == '__main__':
    arguments = docopt(__doc__)
    main(arguments)
