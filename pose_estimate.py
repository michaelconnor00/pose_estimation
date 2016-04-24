import math
import numpy as np
import cv2
from marker_identification import create_rect


def draw_ucs(img, contour):
    """
    Function to draw a 3D coordinate system on the image.
    img - image to draw the ucs on.
    contour - the contour for the marker.
    """
    dst, rect, maxWidth, maxHeight = create_rect(contour)

    opp_pts_long = np.zeros((2,2), dtype="float32")
    opp_pts_long[0], opp_pts_long[1] = get_opp_pts_long(rect)

    opp_pts_short = np.zeros((2,2), dtype="float32")
    opp_pts_short[0], opp_pts_short[1] = get_opp_pts_short(rect)

    cv2.line(img, tuple(opp_pts_long[0]), tuple(opp_pts_long[1]), (255,0,0), 5)
    cv2.line(img, tuple(opp_pts_short[0]), tuple(opp_pts_short[1]), (0,0,255), 5)
    show(img)

    # Get the point with the minimum x value, consider it to be the origin.
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
    # cv2.line(img, tuple(origin), tuple(ucs_coords[0]), (0,255,0), 5)
    # cv2.line(img, tuple(origin), tuple(ucs_coords[1]), (0,255,0), 5)
    # show(img)


def get_2d_points(rect, origin):
    """
    This function finds the distance between the origin, and all the points in
    the list. A distance of zero is ignored, and the longest distance is removed.
    Leaving only the two neighbor points.
    rect - array of marker boundary points
    origin - the point chosen as the origin
    """
    point_dict = []
    dist_list = []

    # Calcualte all distances and add to list.
    for point in rect:
        dist = pts_dist(origin, point)
        if dist > 0:
            dist_list.append(dist)
            point_dict.append({
                "pt": point,
                "dist": dist
            })

    # Remove longest dist form the list.
    dist_list.sort()
    dist_list = dist_list[:2]
    pt_list = []
    for pt in point_dict:
        if pt['dist'] in dist_list:
            pt_list.append(pt['pt'])

    return pt_list


def pts_dist(org, pt):
    """
    Return the distance between the 2 points
    """
    return math.sqrt(
        math.pow(pt[0] - org[0], 2) + math.pow(pt[1] - org[1], 2)
    )


def get_list_of_distances(corners):
    """
    Gets a list of distances between a point and it's neighbor,
    and it's neighbors neighbor (ie - i to i+1, i to i+2)
    corners - array of 4 corners
    """
    distances = []

    for i in range(len(corners)):

        if i < len(corners)-2:
            # Normal
            neighbor = i+1
            neighbor_neighbor = i+2
        elif i == len(corners)-2:
            neighbor = i+1
            neighbor_neighbor = 0
        elif i == len(corners)-1:
            neighbor = 0
            neighbor_neighbor = 1

        distances.append({
            "start_pt": corners[i],
            "end_pt": corners[neighbor_neighbor],
            "dist": pts_dist(corners[i], corners[neighbor_neighbor])
        })

        distances.append({
            "start_pt": corners[i],
            "end_pt": corners[neighbor],
            "dist": pts_dist(corners[i], corners[neighbor]),
        })

    return distances


def get_diag_dist(distances):
    """
    Takes an array of dictionaries describing distance between points
    of which contains duplicates. The duplicates identifies the edges
    that are diagonal, because they will be repeated in the get_list_of_distances() func.
    """
    diag_dists = []
    for dist in distances:
        count = 0
        for d in distances:
            if isclose(dist['dist'], d['dist']):
                count += 1
        if count == 2:
            diag_dists.append(dist)

    de_dup = []

    for dist in diag_dists:
        not_found = True
        for d in de_dup:
            if dist['dist'] == d['dist']:
                not_found = False
        if not_found:
            de_dup.append(dist)
    return de_dup


def get_opp_pts_long(corners):
    """
    Function returns the opposite points for the longest diagonal line.
    """
    distances = get_list_of_distances(corners)

    diagonal_distances = get_diag_dist(distances)

    largest_dist = None
    for dist in diagonal_distances:
        found_larger = False
        for d in diagonal_distances:
            if d['dist'] > dist['dist']:
                found_larger = True
        if found_larger is False:
            largest_dist = dist
    print "longest: ", largest_dist['dist'], ' ALL: ', [d['dist'] for d in distances]
    return largest_dist['start_pt'], largest_dist['end_pt']


def get_opp_pts_short(corners):
    """
    Function returns the opposite points for the shortest diagonal line.
    """
    distances = get_list_of_distances(corners)

    diagonal_distances = get_diag_dist(distances)

    shortest_dist = None
    for dist in diagonal_distances:
        found_shorter = False
        for d in diagonal_distances:
            if d['dist'] == 0.0:
                continue
            if d['dist'] < dist['dist']:
                found_shorter = True
        if found_shorter is False:
            shortest_dist = dist
    print "shortest: ", shortest_dist['dist']
    return shortest_dist['start_pt'], shortest_dist['end_pt']


def show(img):
    cv2.imshow('image',img)
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()


def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)
