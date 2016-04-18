"""
Usage:
    app.py [<file_name>]

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


    # img = draw(img,corners2,imgpts)
    # cv2.imshow('img',img)
    # k = cv2.waitKey(0) & 0xff
    # if k == 's':
    #     cv2.imwrite(fname[:6]+'.png', img)


# def draw(img, corners, imgpts):
#     """
#     Function from OpenCV tutorial: http://docs.opencv.org/3.1.0/d7/d53/tutorial_py_pose.html#gsc.tab=0
#     """
#     corner = tuple(corners[0].ravel())
#     img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
#     img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
#     img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
#     return img


if __name__ == '__main__':
    arguments = docopt(__doc__)
    main(arguments)
