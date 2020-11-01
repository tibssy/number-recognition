import numpy as np
import cv2
import onr
import os
import stfilter


def zoom(frame):
    frame = frame[zoom_y:zoom_y+zoom_h, zoom_x:zoom_x+zoom_w]
    return cv2.resize(frame, (resolution[0], resolution[1]))


def prepare_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return cv2.medianBlur(thresh, 13)


cap = cv2.VideoCapture(0)
resolution = np.flip(np.asarray(cap.read()[1].shape)[:2])
zoom_x, zoom_y = np.int_(resolution / 4)
zoom_w, zoom_h = np.int_(resolution / 2)


while(True):
    num = ""
    accuracy = ""
    ret, frame = cap.read()
    frame = zoom(frame)
    median = prepare_frame(frame)
    median = onr.auto_inv(median)

    stats = cv2.connectedComponentsWithStats(median, connectivity=8)[2][1:]

    filtered = stfilter.filter(stats, resolution)

    for i in filtered:
        x, y, w, h, size = i
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        sgmt = 255 - median[y:y+h, x:x+w]
        data = onr.detect(sgmt)
        num += str(data[0])
        acc = str(data[1])
        accuracy += acc + " "

    os.system("clear")
    print("The number is:", num, "\naccuracy: ", accuracy, "%")

    cv2.imshow('median', median)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
