from imutils.video import VideoStream
import argparse
import imutils
import time
import cv2
import sys
import numpy as np

ap = argparse.ArgumentParser()
# ap.add_argument("-i","--image", required=True,
#         help="path to input image containing ArUCo tag")
ap.add_argument("-t","--type", type = str,
        default="DICT_APRILTAG_36h11",
        help = "type of ArUCo tag to detect")
args = vars(ap.parse_args())

ARUCO_DICT = {
	"DICT_4X4_50": cv2.aruco.DICT_4X4_50,
	"DICT_4X4_100": cv2.aruco.DICT_4X4_100,
	"DICT_4X4_250": cv2.aruco.DICT_4X4_250,
	"DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
	"DICT_5X5_50": cv2.aruco.DICT_5X5_50,
	"DICT_5X5_100": cv2.aruco.DICT_5X5_100,
	"DICT_5X5_250": cv2.aruco.DICT_5X5_250,
	"DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
	"DICT_6X6_50": cv2.aruco.DICT_6X6_50,
	"DICT_6X6_100": cv2.aruco.DICT_6X6_100,
	"DICT_6X6_250": cv2.aruco.DICT_6X6_250,
	"DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
	"DICT_7X7_50": cv2.aruco.DICT_7X7_50,
	"DICT_7X7_100": cv2.aruco.DICT_7X7_100,
	"DICT_7X7_250": cv2.aruco.DICT_7X7_250,
	"DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
	"DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
	"DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
	"DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
	"DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
	"DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
}

# # load the input image from disk and resize it
# print("[INFO] loading image...")
# image = cv2.imread(args["image"])
# image = imutils.resize(image, width = 600)

#verify that the supplied ArUCo tag exists and is supported by
#OpenCV
# if ARUCO_DICT.get(args["type"],None) is None:
#     print("[INFO] ArUCo tag of '{}' is not supported".format(
#                 args["type"]))
#     sys.exit(0)

# load the ArUCo dictionary, grab the ArUCo parameters, and detect
# the markers

print("[INFO] detecting '{}' tags...".format(args["type"]))
arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT[args["type"]])
arucoParams = cv2.aruco.DetectorParameters_create()
# (corners, ids, rejected) = cv2.aruco.detectMarkers(image, arucoDict,
# 	parameters=arucoParams)



#initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

#Camera Settings

aruco_len =  0.163 # 16.3 cm
camera_matrix = np.matrix([
[ 757.7574154293701, 0, 638.4948152909574],
[ 0, 758.5212504108516, 406.45129752550054],
[ 0, 0, 1] ])

camera_dist = np.array([ 0.1295301484374958, -0.1325476683656682, 0.0029232886276809892, -0.003878180299787567, 0.021085953642091773 ])
#loop over the frames from the video stream
while True:

    #grab the frame from the threaded video stream and resize it
    #to have a max width of 1000 pixels
    frame = vs.read()
    frame = imutils.resize(frame,width = 1000)
    #print("after resize")

    #detect ArUco Markers in the input frame
    (corners,ids, rejected) = cv2.aruco.detectMarkers(frame,arucoDict,parameters = arucoParams)
    #print("detect ArUco")
    '''


    The cv2.aruco.detectMarkers results in a 3-tuple of:

        corners: The (x, y)-coordinates of our detected ArUco markers
        ids: The identifiers of the ArUco markers (i.e., the ID encoded in the marker itself)
        rejected: A list of potential markers that were detected but ultimately rejected due to the code inside the marker not being able to be parsed


    '''


    # verify *at least* one ArUco marker was detected


    if len(corners) > 0:
        # flatten the ArUco IDs list
        #print("id flattening")
        ids = ids.flatten()
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners[0], aruco_len, camera_matrix, camera_dist)
        see_rvecs = rvecs*100
        see_tvecs = tvecs*100
        #print(see_tvecs, see_rvecs)
        #Define 3*3 matrix for rotation
        # rot_mat = np.array([[0,0,0],[0,0,0],[0,0,0]])
        rot_mat = np.zeros([3,3])

        cv2.Rodrigues(rvecs,rot_mat)

        print(rot_mat)

        # loop over the detected ArUCo corners
        for (markerCorner, markerID) in zip(corners, ids):
            # extract the marker corners (which are always returned in
            # top-left, top-right, bottom-right, and bottom-left order)
            corners = markerCorner.reshape((4, 2))
            (topLeft, topRight, bottomRight, bottomLeft) = corners

            # convert each of the (x, y)-coordinate pairs to integers
            topRight = (int(topRight[0]), int(topRight[1]))
            bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
            bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
            topLeft = (int(topLeft[0]), int(topLeft[1]))



            # draw the bounding box of the ArUCo detection
            cv2.line(frame, topLeft, topRight, (0, 255, 0), 2)
            cv2.line(frame, topRight, bottomRight, (0, 255, 0), 2)
            cv2.line(frame, bottomRight, bottomLeft, (0, 255, 0), 2)
            cv2.line(frame, bottomLeft, topLeft, (0, 255, 0), 2)

            # compute and draw the center (x, y)-coordinates of the ArUco
            # marker
            cX = int((topLeft[0] + bottomRight[0]) / 2.0)
            cY = int((topLeft[1] + bottomRight[1]) / 2.0)
            cv2.circle(frame, (cX, cY), 4, (0, 0, 255), -1)

        # draw the ArUco marker ID on the image
        cv2.putText(frame, str(markerID),
            (topLeft[0], topLeft[1] - 15), cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (0, 255, 0), 2)
        cv2.drawFrameAxes(frame,camera_matrix,camera_dist,rvecs,tvecs,0.1,3)
        #print("[INFO] ArUco marker ID: {}".format(markerID))

		# show the output image
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF


        #if the 'q' key was pressed, break from the loop
        if key == ord('q'):
            break

# do a bit of cleanup here
cv2.destroyAllWindows()
vs.stop()


# camera calinration params

# {
#     "camera": "HP Wide Vision HD (04f2:b56d)",
#     "platform": "X11; Linux x86_64",
#     "camera_matrix": [
#         [
#             757.7574154293701,
#             0,
#             638.4948152909574
#         ],
#         [
#             0,
#             758.5212504108516,
#             406.45129752550054
#         ],
#         [
#             0,
#             0,
#             1
#         ]
#     ],
#     "distortion_coefficients": [
#         0.1295301484374958,
#         -0.1325476683656682,
#         0.0029232886276809892,
#         -0.003878180299787567,
#         0.021085953642091773
#     ],
#     "distortion_model": "rectilinear",
#     "avg_reprojection_error": 0.16253617870389453,
#     "img_size": [
#         1280,
#         720
#     ],
#     "keyframes": 12,
#     "calibration_time": "Sat, 13 Aug 2022 17:25:10 GMT"
# }
