import threading
import cv2
import numpy as np
import pyaudio
import wave
import foreground_extractor

VIDEO_NAME = 'Test.mp4'
SOUND_NAME = 'alarmz.wav'

SHOW_ORIGIN = 1
SHOW_FOREGROUND = 1
SHOW_MHI = 1
SHOW_COEFFICIENT = 1
PLAY_SOUND = 1


DURATION = 20   # Number of history images to store

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

def alert():
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

def calculateMovementCoefficient(foreground, timestamp):
    global mhi
    step = 0.05     # Image is brighter with smaller step
    w, h = np.shape(foreground)
    if timestamp == 0:
        mhi = np.zeros((w, h), np.float32)
        return 0

    foreground[ foreground > 0 ] = 1
    fgsum = foreground.sum()
    if fgsum == 0: return 0

    mhi = mhi - step
    mhi[ foreground != 0 ] = 1
    mhi[ (mhi < 1 - DURATION * step) & (foreground == 0) ] = 0
    if SHOW_MHI: cv2.imshow('Motion history', mhi)
    return mhi.sum() / fgsum

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

while True:
    # Press q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'): break
    hasNext, frame = cap.read()
    if not hasNext: continue

    foreground = findForeground(frame)
    MC = calculateMovementCoefficient(foreground, count)

    maxCnt = findMaxContour(foreground)
    if maxCnt is not None:
        (x, y), (a, b), angle = ellipse = cv2.fitEllipse(maxCnt)

        cv2.ellipse(frame, ellipse, (0, 255, 0), 2)

    if SHOW_ORIGIN: cv2.imshow('frame', frame)
    if SHOW_COEFFICIENT: print "Frame %d:, MC: %.2f" % (count, MC)
    count += 1

cap.release()
cv2.destroyAllWindows()


