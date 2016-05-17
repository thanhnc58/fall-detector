import threading
import cv2
import numpy as np
import fall_detector
import pyaudio
import wave
import foreground_extractor

dataset_url = './'

def beep(sound):
    chunk = 1024

    f = wave.open(r"%s.wav" % sound,"rb")
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

count = 1
cap = cv2.VideoCapture(dataset_url + 'video (' + str(count) + ').avi')

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6, 6))
reset = False
positive = False

while 1:
    ret, frame = cap.read()
    if frame is None:
        count += 1
        cap = cv2.VideoCapture(dataset_url + 'video (' + str(count) + ').avi')
        reset = True
        foreground_extractor.reset()
        cv2.waitKey(300)
        continue
    post_processing = frame
    post_processing = cv2.medianBlur(post_processing, 7)
    foreGround = foreground_extractor.mog2(post_processing)
    ret, foreground_binary = cv2.threshold(foreGround, 230, 255, cv2.THRESH_BINARY)
    cv2.imshow('post_processing', post_processing)
    cv2.imshow('foreground', foreground_binary)

    ###############################################################

    image, contours, hierarchy = cv2.findContours(foreground_binary, 1, 2)

    if (len(contours) > 0):
        maxCnt = contours[0]
        for cnt in contours:
            a = cv2.contourArea(cnt)
            if a > cv2.contourArea(maxCnt):
                maxCnt = cnt

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
            topmost = tuple(maxCnt[maxCnt[:, :, 1].argmin()][0])
            bottommost = tuple(maxCnt[maxCnt[:, :, 1].argmax()][0])

            fall = fall_detector.predict(FA, AR, HP, MC, reset)
            if fall == 1:
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, 'OOPS!', (100, 100), font, 1, (0, 0, 255), 2, cv2.LINE_AA)

                t = threading.Thread(target=beep, args=('alarmz',))
                t.start()

            reset = False

    cv2.imshow('frame', frame)

    ###############################################################

    positive = False

    k = cv2.waitKey(12) & 0xff
    if k != 255:
        print(k)
    if k == 112:
        positive = True
        k = cv2.waitKey() & 0xff
    elif k == 27:
        break

cap.release()
cv2.destroyAllWindows()
