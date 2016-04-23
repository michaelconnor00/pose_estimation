"""
This file contains some snippets of previously used code that is not needed,
but being kep for reference.
"""
import os
import numpy as np
import cv2
from docopt import docopt

def main(arguments):
    image_filename = arguments['<file_name>']

    cwd = os.getcwd()

    img = cv2.imread(os.path.join(cwd, image_filename))

    # http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_feature2d/py_shi_tomasi/py_shi_tomasi.html
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    thresh, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    kernel = np.ones((5,5),np.uint8)

    erosion = cv2.erode(binary, kernel, iterations = 1)
    sub_img = cv2.subtract(binary, erosion)

    # show(sub_img)

    corners = cv2.goodFeaturesToTrack(sub_img, 30, 0.01, 10)
    corners = np.int0(corners)

    # show(cv2.morphologyEx(binary, cv2.MORPH_BLACKHAT, kernel))

    # corners = cv2.boundingRect(corners)
    # print len(corners), type(corners[0]), corners


    for i in corners:
        x, y = i.ravel()
        cv2.circle(img, (x,y), 3, (0,0,255), -1)

    show(img)

def show(img):
    cv2.imshow('image',img)
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()


def draw(img, corners, imgpts):
    """
    Function from OpenCV tutorial: http://docs.opencv.org/3.1.0/d7/d53/tutorial_py_pose.html#gsc.tab=0
    """
    corner = tuple(corners[0].ravel())
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img


if __name__ == '__main__':
    arguments = docopt(__doc__)
    main(arguments)

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
