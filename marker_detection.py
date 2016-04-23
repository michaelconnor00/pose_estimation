import numpy as np
import cv2


def check_for_id(gray, contour):
    warp = get_warped_img(gray, contour)

    # show(draw_grid(warp))

    # a 3x3 array of bits to ID the marker.
    id_bits = get_id_bits(warp)

    # print id_bits

    return validate_id(id_bits)


def get_warped_img(gray, contour):
    # Create a rectangle of points and a destination points array
    dst, rect, maxWidth, maxHeight = create_rect(contour)

    # calculate the perspective transform matrix and warp
    # the perspective to grab the screen
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(gray, M, (maxWidth, maxHeight))


def create_rect(contour):
    # Create array of zeros
    rect = np.zeros((4, 2), dtype="float32")

    # reshape the contour array to a 4x2
    pts = contour.reshape(4, 2)

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

    return dst, rect, maxWidth, maxHeight


def validate_id(bits_array):
    is_valid = True

    if len(bits_array) != 3:
        is_valid = False

    for row in bits_array:
        if len(row) != 3:
            is_valid = False

        for bit in row:
            if bit not in [1, 0]:
                is_valid = False

    return is_valid


def get_id_bits(img):
    x_offset = img.shape[1]/5
    y_offset = img.shape[0]/5

    id_bits = [[],[],[]]

    for i in range(3):
        for j in range(3):
            bit = extract_bit(
                img,
                y_offset*(i+1), y_offset*(i+2),
                x_offset*(j+1), x_offset*(j+2)
            )
            id_bits[i].append(bit)
    return id_bits


def extract_bit(img, y_start, y_end, x_start, x_end):
    # constant for ignoring pixels around the boarder
    ignore_percent = 0.1
    const = int(((img.shape[0]/5 + img.shape[1]/5)/2) * ignore_percent)
    y_start += const; x_start+= const
    y_end -= const; x_end -= const

    # subrect of an image: img[y1:y2, x1:x2]
    subrect = img[y_start:y_end, x_start:x_end]
    # show(subrect)
    _, thresh = cv2.threshold(subrect, 128, 255, cv2.THRESH_BINARY)
    total_pixels = thresh.size
    non_zero_pixels = cv2.countNonZero(thresh)
    # print 'NON/TOTAL: ', non_zero_pixels, total_pixels
    percentage_non_zero = float(non_zero_pixels)/float(total_pixels)
    # print 'PER:', percentage_non_zero
    if percentage_non_zero < 0.5:
        return 0
    elif percentage_non_zero >= 0.5:
        return 1


def draw_grid(img):
    img_copy = img.copy()

    x_offset = img.shape[1]/5
    y_offset = img.shape[0]/5

    # Vert lines
    for i in range(4):
        cv2.line(img_copy, (x_offset*(i+1), y_offset*0), (x_offset*(i+1), y_offset*5), (255,0,0), 2)

    # Horiz lines
    for i in range(4):
        cv2.line(img_copy, (x_offset*0, y_offset*(i+1)), (x_offset*5, y_offset*(i+1)), (255,0,0), 2)

    return img_copy


# def show(img):
#     cv2.imshow('image',img)
#     if cv2.waitKey(0) & 0xff == 27:
#         cv2.destroyAllWindows()
