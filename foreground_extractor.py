import numpy as np
import cv2

fgbg_mog2 = cv2.createBackgroundSubtractorMOG2(500, 60, True)
fgbg_mog = cv2.bgsegm.createBackgroundSubtractorMOG(500, 20)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (14, 14))

def reset():
    global fgbg_mog
    global fgbg_mog2
    global kernel
    fgbg_mog2 = cv2.createBackgroundSubtractorMOG2(500, 60, True)
    fgbg_mog = cv2.bgsegm.createBackgroundSubtractorMOG(500, 20)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (14, 14))

def mog2(img):
    fgmask = fgbg_mog2.apply(img)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_DILATE, kernel)
    return fgmask


def mog(img):
    fgmask = fgbg_mog.apply(img)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    return fgmask