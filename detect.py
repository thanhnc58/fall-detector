import threading
import cv2
import numpy as np
import pyaudio
import wave
import foreground_extractor

VIDEO_NAME = 'Test.mp4'
SOUND_NAME = 'alarmz.wav'

SHOW_ORIGIN = 1
SHOW_FOREGROUND = 0
SHOW_MHI = 0
SHOW_COEFFICIENT = 1
PLAY_SOUND = 1

STEP_BY_STEP = 1                # Press c to move to next frame
FRAME_SKIP = 0                  # Number of frames to skip when showing frames step by step

MOTION_DURATION = 20            # Number of history images to store
MOTION_HISTORY_STEP = 0.01      # Motion history image is brighter with smaller step
MOVEMENT_COEFFICIENT = 1.2      # Maximum movement coefficient, above which system alerts


ANGLE_DURATION = 5
ANGLE_STANDARD_DEVIATION = 15   # Maximum ellipse angle standard deviation, above which system alerts

RATIO_DURATION = 5
RATIO_STANDARD_DEVIATION = 0.9  # Maximum standard deviation of ratio of major and minor axes of ellipse, above which system alerts

count = 0
mhi = None
angleList = []
ratioList = []

def beep():
    chunk = 1024
    f = wave.open(r"%s" % SOUND_NAME,"rb")
    p = pyaudio.PyAudio()
    stream = p.open(format = p.get_format_from_width(f.getsampwidth()),
                    channels = f.getnchannels(),
                    rate = f.getframerate(),
                    output = True)

    data = f.readframes(chunk)
    while data != '':
        stream.write(data)
        data = f.readframes(chunk)

    stream.stop_stream()
    stream.close()
    p.terminate()

def alert(frame):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, 'OOPS!', (100, 100), font, 1, (0, 0, 255), 2, cv2.LINE_AA)

    if PLAY_SOUND and threading.activeCount() == 1:
        t = threading.Thread(target=beep)
        t.start()

def findForeground(frame):
    blured = cv2.medianBlur(frame, 7)
    foreground = foreground_extractor.mog2(blured)
    ret, foregroundBinary = cv2.threshold(foreground, 230, 255, cv2.THRESH_BINARY)
    if SHOW_FOREGROUND: cv2.imshow('foreground', foregroundBinary)
    return foregroundBinary

def findMaxContour(foreground):
    image, contours, hierarchy = cv2.findContours(foreground, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0: return None

    maxCnt = contours[0]
    maxArea = cv2.contourArea(maxCnt)
    for cnt in contours:
        a = cv2.contourArea(cnt)
        if a > maxArea:
            maxCnt = cnt
            maxArea = a
    perimeter = cv2.arcLength(maxCnt, True)
    if perimeter < 200: return None
    return maxCnt

def calculateMovementCoefficient(foreground, timestamp):
    global mhi
    w, h = np.shape(foreground)
    if timestamp == 0:
        mhi = np.zeros((w, h), np.float32)
        return 0

    foreground[ foreground > 0 ] = 1
    fgsum = foreground.sum()
    if fgsum == 0: return 0

    mhi = mhi - MOTION_HISTORY_STEP
    mhi[ foreground != 0 ] = 1
    mhi[ (mhi < 1 - MOTION_DURATION * MOTION_HISTORY_STEP) & (foreground == 0) ] = 0
    if SHOW_MHI: cv2.imshow('Motion history', mhi)
    return mhi.sum() / fgsum

def calculateAngleStandardDeviation(ellipse):
    (x, y), (a, b), angle = ellipse
    if angle > 90: angle = 180 - angle
    angleList.append(angle)
    npAngleList = np.array(angleList[-1-ANGLE_DURATION:-1])
    return np.std(npAngleList)

def calculateRatioStandardDeviation(ellipse):
    (x, y), (a, b), angle = ellipse
    ratio = a / b
    ratioList.append(ratio)
    npRatioList = np.array(ratioList[-1-RATIO_DURATION:-1])
    return np.std(npRatioList)

def fallDetected(MC, AD, RD):
    if SHOW_COEFFICIENT:
        print "Frame %d:, MC: %.2f, AD: %.2f, RD: %.2f" % (count, MC, AD, RD)
    return  MC > MOVEMENT_COEFFICIENT and \
            AD > ANGLE_STANDARD_DEVIATION and \
            RD > RATIO_STANDARD_DEVIATION

def analysis():
    global count
    cap = cv2.VideoCapture(VIDEO_NAME)

    while True:
        # Press q to quit
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'): break
        if STEP_BY_STEP and count > FRAME_SKIP and k != ord('c'): continue

        hasNext, frame = cap.read()
        if not hasNext: continue

        foreground = findForeground(frame)
        MC = calculateMovementCoefficient(foreground, count)
        maxCnt = findMaxContour(foreground)

        if maxCnt is not None:
            (x, y), (a, b), angle = ellipse = cv2.fitEllipse(maxCnt)
            cv2.ellipse(frame, ellipse, (0, 255, 0), 2)
            AD = calculateAngleStandardDeviation(ellipse)
            RD = calculateRatioStandardDeviation(ellipse)
            if fallDetected(MC, AD, RD): alert(frame)

        if SHOW_ORIGIN: cv2.imshow('frame', frame)
        count += 1

    cap.release()
    cv2.destroyAllWindows()

analysis()
