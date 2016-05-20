import threading
import cv2
import numpy as np
import fall_detector
import pyaudio
import wave
import foreground_extractor

VIDEO_NAME = 'Test.mp4'
SOUND_NAME = 'alarmz.wav'

SHOW_ORIGIN = 0
SHOW_FOREGROUND = 1
SHOW_MHI = 1

mhi = None
count = 0
cap = cv2.VideoCapture(VIDEO_NAME)

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

def postProcess(frame):
    blured = cv2.medianBlur(frame, 7)
    foreground = foreground_extractor.mog2(blured)
    ret, foregroundBinary = cv2.threshold(foreground, 230, 255, cv2.THRESH_BINARY)
    if SHOW_FOREGROUND: cv2.imshow('foreground', foregroundBinary)
    return foregroundBinary

def calculateMovementCoefficient(foreground, timestamp):
    global mhi
    duration = 20   #Number of history images to store
    step = 0.05     #Image is brighter with smaller step
    w, h = np.shape(foreground)
    if timestamp == 0:
        mhi = np.zeros((w, h), np.float32)
        return 0

    foreground[ foreground > 0 ] = 1
    fgsum = foreground.sum()
    if fgsum == 0: return 0

    mhi = mhi - step
    mhi[ foreground != 0 ] = 1
    mhi[ (mhi < 1 - duration * step) & (foreground == 0) ] = 0
    if SHOW_MHI: cv2.imshow('Motion history', mhi)
    return mhi.sum() / fgsum

def findMaxContour(frame):
    image, contours, hierarchy = cv2.findContours(processedFrame, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0: return None

    maxCnt = contours[0]
    maxArea = cv2.contourArea(maxCnt)
    for cnt in contours:
        a = cv2.contourArea(cnt)
        if a > maxArea:
            maxCnt = cnt
            maxArea = a
    return maxCnt

while True:
    if cv2.waitKey(1) & 0xFF == ord('q'): break
    hasNext, frame = cap.read()
    if not hasNext: continue

    processedFrame = postProcess(frame)
    MC = calculateMovementCoefficient(processedFrame, count)
    print MC
    maxCnt = findMaxContour(processedFrame)
    if maxCnt is not None:
        perimeter = cv2.arcLength(maxCnt, True)
        if perimeter > 200:
            # ellipse = ((0, 0), (0, 0), 0)
            rect = cv2.minAreaRect(maxCnt)
            (x, y), (w, h), rect_angle = rect
            # The output of cv2.minAreaRect() is ((x, y), (w, h), angle).
            # Using cv2.cv.BoxPoints() is meant to convert this to points
            # https://namkeenman.wordpress.com/2015/12/18/open-cv-determine-angle-of-rotatedrect-minarearect/

            # cv2.circle(frame, (int(x),int(y)), 2, (0, 255, 255), 2)
            myContourAngle = rect_angle
            if w < h:
                myContourAngle = -(myContourAngle - 90)
            FA = myContourAngle
            # print('FA: ' + str(FA))

            box = cv2.boxPoints(rect)
            box = np.int0(box)
            # print(box)

            if w < h:
                AR = w / h
                HP = ((box[1] + box[2]) / 2).astype(int)
            else:
                AR = h / w
                HP = ((box[0] + box[1]) / 2).astype(int)

            # print('AR: ' + str(AR))

            # print('HP: ' + str(HP))

            (x, y), (MA, ma), angle = cv2.fitEllipse(maxCnt)
            ellipse = (x, y), (MA, ma), angle
            # https://namkeenman.wordpress.com/2015/12/21/opencv-determine-orientation-of-ellipserotatedrect-in-fitellipse/
            cv2.ellipse(frame, ellipse, (0, 255, 0), 2)
            cv2.drawContours(frame, [box], 0, (255, 0, 0), 2)
            cv2.circle(frame, tuple(HP), 2, (0, 255, 255), 2)
            # print('angle: ' + str(angle))

            # Mass center
            # http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_imgproc/py_contours/py_contour_features/py_contour_features.html
            M = cv2.moments(maxCnt)
            centroid_x = int(M['m10'] / M['m00'])
            centroid_y = int(M['m01'] / M['m00'])
            cv2.circle(frame, (centroid_x, centroid_y), 2, (0, 0, 255), 3)

            MC = (centroid_x, centroid_y)
            # print('MC: ' + str(MC))

            # http://opencvpython.blogspot.com/2012/06/hi-this-article-is-tutorial-which-try.html
            # http://opencvpython.blogspot.com/2012/06/contours-2-brotherhood.html
            # http://opencvpython.blogspot.com/2012/06/contours-3-extraction.html
            # topmost = tuple(maxCnt[maxCnt[:, :, 1].argmin()][0])
            # bottommost = tuple(maxCnt[maxCnt[:, :, 1].argmax()][0])

            fall = fall_detector.predict(FA, AR, HP, MC, False)
            if fall == 1:
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, 'OOPS!', (100, 100), font, 1, (0, 0, 255), 2, cv2.LINE_AA)

                if threading.activeCount() == 1:
                    t = threading.Thread(target=beep)
                    t.start()

    if SHOW_ORIGIN: cv2.imshow('frame', frame)
    count += 1

    ###############################################################

cap.release()
cv2.destroyAllWindows()


