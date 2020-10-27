import numpy as np
import cv2
import engine
import os

max_char = 10


def filter(stats):
    stats = stats[np.prod(stats[:, :2], axis=1) != 0]
    sorted_stats = stats[np.argsort(stats[:, 0])][:max_char]
    return sorted_stats


def zoom(frame):
    frame = frame[zoom_y:zoom_y+zoom_h, zoom_x:zoom_x+zoom_w]
    return cv2.resize(frame, (resolution[0], resolution[1]))


def prepare_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return cv2.medianBlur(thresh, 5)


def auto_inv(median):
    if np.round(100 - np.sum(median / 255) / (median.shape[0] * median.shape[1]) * 100).astype(int) < 50:
        return 255 - median
    return median


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
    median = auto_inv(median)

    stats = cv2.connectedComponentsWithStats(median, connectivity=8)[2][1:]

    filtered = filter(stats)

    for i in filtered:
        x, y, w, h, size = i
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        sgmt = 255 - median[y:y+h, x:x+w]
        data = engine.detect(sgmt)
        num += str(data[0])
        acc = str(data[2])
        accuracy += acc + " "

    os.system("clear")
    print("The number is:", num, "\naccuracy: ", accuracy, "%")

 #   print(filtered)

    cv2.imshow('median', median)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
