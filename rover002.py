import numpy as np
from utils import ARUCO_DICT, aruco_display
import time
import argparse
import cv2
import sys

'''
Sample Command:-
python .py --image marker_1.png --type DICT_5X5_100
'''

ap = argparse.ArgumentParser()
ap.add_argument("-D:\\rover\\rovv\\marker_1.png", required=True, help="Path to input image containing ArUco tag")
ap.add_argument("-DICT_5X5_100", type=str, default="DICT_ARUCO_ORIGINAL", help="Type of ArUco tag to detect")
args = vars(ap.parse_args())
image_path = args["image"]

# Load the image
image = cv2.imread(image_path)


print("Loading image...")
image = cv2.imread(args["image"])
h,w,_ = image.shape
width=600
height = int(width*(h/w))
image = cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)


# verify that the supplied ArUCo tag exists and is supported by OpenCV
if ARUCO_DICT.get(args["type"], None) is None:
	print(f"ArUCo tag type '{args['type']}' is not supported")
	sys.exit(0)

# load the ArUCo dictionary, grab the ArUCo parameters, and detect
# the markers
print("Detecting '{}' tags....".format(args["type"]))
arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT[args["type"]])
arucoParams = cv2.aruco.DetectorParameters_create()
corners, ids, rejected = cv2.aruco.detectMarkers(image, arucoDict, parameters=arucoParams)

detected_markers = aruco_display(corners, ids, rejected, image)
cv2.imshow("Image", detected_markers)

# # Uncomment to save
# cv2.imwrite("output_sample.png",detected_markers)

cv2.waitKey(0)