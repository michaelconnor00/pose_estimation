import os
import math
import numpy as np
import cv2
from utils import create_rect


def draw_ucs(img, contour):
    print contour
    dst, rect, maxWidth, maxHeight = create_rect(contour)

    min_x = None
    for point in rect:
        if min_x is None:
            min_x = point
        if point[0] < min_x[0]:
            min_x = point


    origin = np.zeros((2), dtype="float32")
    origin[0] = min_x[0]; origin[1] = min_x[1]

    # Find X and Y
    ucs_coords = np.zeros((3,2), dtype="float32")
    ucs_coords[0], ucs_coords[1] = get_2d_points(rect, origin)
    print ucs_coords

    ucs_coords = get_z_coord(origin, ucs_coords)

    # matrix = cv2.getPerspectiveTransform(rect, dst)
    # print 'Matrix: ', matrix

    # return draw_rect(img, rect)

def get_z_coord(origin, ucs_coords):
    


def get_2d_points(rect, origin):
    point_dict = []
    dist_list = []
    for point in rect:
        dist = pts_dist(origin, point)
        if dist > 0:
            dist_list.append(dist)
            point_dict.append({
                "pt": point,
                "dist": dist
            })

    # Remove longest dist
    dist_list.sort()
    dist_list = dist_list[:2]
    pt_list = []
    for pt in point_dict:
        if pt['dist'] in dist_list:
            pt_list.append(pt['pt'])

    return pt_list


def pts_dist(org, pt):
    return math.sqrt(
        math.pow(pt[0] - org[0], 2) + math.pow(pt[1] - org[1], 2)
    )


def draw_coord(img, origin, x_pt, y_pt, z_pt):
    cv2.line(img, tuple(origin), tuple(x_pt), (255,0,0), 5)
    cv2.line(img, tuple(origin), tuple(y_pt), (0,255,0), 5)
    cv2.line(img, tuple(origin), tuple(z_pt), (0,0,255), 5)
    return img


def show(img):
    cv2.imshow('image',img)
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()
