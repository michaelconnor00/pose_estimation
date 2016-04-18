"""
Usage:
    3D_2Dpose.py [<file_name>]

Script for estimating 3D coordinates of a marker from a 2D image
and the marker resource.

Arguments:
    <file_name>     The name of the image file

Options:
    --verbose       Print debug output
    --show          Show output

"""
import os
import numpy as np
import cv2
from docopt import docopt

def main(arguments):
    print arguments
    image_filename = arguments['<file_name>']
    show_output = arguments['--show']

    cwd = os.getcwd()

    img = cv2.imread(os.path.join(cwd, image_filename))

    # http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_feature2d/py_shi_tomasi/py_shi_tomasi.html
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    corners = cv2.goodFeaturesToTrack(gray, 25, 0.01, 10)
    corners = np.int0(corners)

    for i in corners:
        x, y = i.ravel()
        cv2.circle(img, (x,y), 3, (0,0,255), -1)

    cv2.imwrite('skewed_w_corners.jpg', img)


    if show_output:
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
