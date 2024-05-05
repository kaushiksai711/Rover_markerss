'''
Sample Command:-
python detect_aruco_images.py --image Images/test_image_1.png --type DICT_5X5_100
python D:\rover\rovv\rover00img.py --image D:\rover\marker_5_page-0001.jpg --type DICT_5X5_100
'''
import numpy as np
from utils import ARUCO_DICT, aruco_display
import argparse
import cv2
import sys


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image containing ArUCo tag")
ap.add_argument("-t", "--type", type=str, default="DICT_ARUCO_ORIGINAL", help="type of ArUCo tag to detect")
args = vars(ap.parse_args())



print("Loading image...")
image = cv2.imread(args["image"])
print(image)
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
arucoDict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[args["type"]])
arucoParams = cv2.aruco.DetectorParameters()


arucoParams.adaptiveThreshWinSizeMin = 3  # Example adjustment
arucoParams.adaptiveThreshWinSizeStep = 1  # Example adjustment
arucoParams.minMarkerPerimeterRate = 0.01  # Example adjustment
# You can adjust more parameters as needed

corners, ids, rejected = cv2.aruco.detectMarkers(image, arucoDict, parameters=arucoParams)
print('sa',corners,'mi',ids,'ag',rejected,'yo')
detected_markers = aruco_display(corners, ids, rejected, image)
cv2.imshow("Image", detected_markers)
cv2.waitKey(0)
'''
a=cv2.aruco.ArucoDetector()
print(a)
corners, ids, rejected_candidates = a.detectMarkers(image)

# Print the detected corners and IDs
print("Detected corners:", corners)
print("Marker IDs:", ids)
print("Rejected candidates:", rejected_candidates)'''
# # Uncomment to save
# cv2.imwrite("output_sample.png",detected_markers)

