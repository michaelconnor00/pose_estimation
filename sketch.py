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
