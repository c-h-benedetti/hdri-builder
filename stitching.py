import imutils as imu
import cv2 as ocv
import numpy as np
import utilities as ut

seq = ut.open_sequence("testing7/", True)
"""
seq = []
seq.append(ocv.imread("testing7/IMG_20200714_170030.jpg"))
seq.append(ocv.imread("testing7/IMG_20200714_170037.jpg"))
seq.append(ocv.imread("testing7/IMG_20200714_170045.jpg"))
seq.append(ocv.imread("testing7/IMG_20200714_170053.jpg"))
"""
stitcher = ocv.createStitcher() if imu.is_cv3() else ocv.Stitcher_create()
(status, stitched) = stitcher.stitch(seq)

if (status == 0):
    ocv.imwrite("stitched.png", stitched)
else:
    print("Failed stitching: " + str(status))