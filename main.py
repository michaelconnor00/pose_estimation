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
from marker_detection import check_for_id
from pose_estimate import draw_ucs


def main(arguments):
    image_filename = arguments['<file_name>']

    cwd = os.getcwd()

    global img
    img = cv2.imread(os.path.join(cwd, image_filename))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # adap_thresh = cv2.adaptiveThreshold( #
    #     img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 23, 2
    # )

    perimeter_contour, contours = get_contours(thresh)

    for contour in contours:
        is_marker = check_for_id(gray, contour)
        if is_marker:
            print 'Marker Found!'
            pose_img = draw_ucs(img, contour)
            # show(pose_img)

def show(img):
    cv2.imshow('image',img)
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()


def get_contours(image):
    """
    Takes a binary image and returns contours filtered by size and squarish shape.
    """
    contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # cv2.drawContours(img, contours, -1, (0,0,255), 3)
    # show(img)

    # The first index is the perimeter of the image, filter it out.
    filtered_contours = contours[1:]

    # Filter the contours by size.
    final_contours = []
    for contour in filtered_contours:
        # Approximate the contour
        perimeter = cv2.arcLength(contour, True)
        # Look for Ploygon shapes
        approx = cv2.approxPolyDP(contour, 0.1 * perimeter, True)

        # Only add to list if contour has 4 corners
        if len(approx) == 4:
            final_contours.append(approx)

    return contours[0], final_contours


if __name__ == '__main__':
    arguments = docopt(__doc__)
    main(arguments)
