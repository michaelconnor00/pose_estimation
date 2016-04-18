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
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # adap_thresh = cv2.adaptiveThreshold( #
    #     img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 23, 2
    # )

    contours = get_contours(thresh)

    # cv2.drawContours(img, contours, -1, (0,0,255), 3)

    # Create array of zeros
    rect = np.zeros((4, 2), dtype="float32")
    print rect

    # reshape the contour array to a 4x2
    pts = contours[0].reshape(4, 2)
    print pts

    # the top-left point has the smallest sum whereas the
    # bottom-right has the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # compute the difference between the points -- the top-right
    # will have the minumum difference and the bottom-left will
    # have the maximum difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # now that we have our rectangle of points, let's compute
    # the width of our new image
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))

    # ...and now for the height of our new image
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))

    # take the maximum of the width and height values to reach
    # our final dimensions
    maxWidth = max(int(widthA), int(widthB))
    maxHeight = max(int(heightA), int(heightB))

    # construct our destination points which will be used to
    # map the screen to a top-down, "birds eye" view
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")

    # calculate the perspective transform matrix and warp
    # the perspective to grab the screen
    M = cv2.getPerspectiveTransform(rect, dst)
    warp = cv2.warpPerspective(img, M, (maxWidth, maxHeight))

    # cv2.drawContours(img, [rect], -1, (0,0,255), 3)

    show(warp)

def show(img):
    cv2.imshow('image',img)
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()

def get_contours(image):
    """
    Takes a binary image and returns contours filtered by size and squarish shape.
    """
    contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # The first index is the perimeter of the image, filter it out.
    filtered_contours = contours[1:]
    print type(filtered_contours)
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

    return final_contours

if __name__ == '__main__':
    arguments = docopt(__doc__)
    main(arguments)
