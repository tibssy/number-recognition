import numpy as np
import cv2

grid_x = 3
grid_y = 3


def threshold_image(gray):
    return cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]


def prepare_image(file):
    img = cv2.imread(file, 0)
    thresh = threshold_image(img)
    stats = cv2.connectedComponentsWithStats(thresh, connectivity=8)[2]
    stats = stats[1]
    x, y, w, h = stats[:4]
    return img[y:y+h, x:x+w]


def resize_image(roi):
    roi_h, roi_w = roi.shape
    while roi_w % grid_x != 0 or roi_h % grid_y != 0:
        if roi_w % grid_x != 0:
            roi_w += 1
        if roi_h % grid_y != 0:
            roi_h += 1

    resized = cv2.resize(roi, (roi_w, roi_h))
    return ((255 - threshold_image(resized)) / 255).astype(int)


def segment_to_data(segment):
    white = np.sum(segment)
    pixel_num = np.prod(np.asanyarray(segment.shape))
    rate = np.round(100 - white / pixel_num * 100)
    return rate


def slice_number(ones):
    res = []
    for i in np.vsplit(ones, grid_y):
        for j in np.hsplit(i, grid_x):
            res.append(segment_to_data(j))

    return np.asanyarray(res).astype(int)


def detect(data):

    def normalize(arr):
        if len(arr.shape) == 1:
            arr -= np.min(arr)
            max_val = np.max(arr)
        else:
            arr -= np.min(arr, axis=1).reshape(arr.shape[0], 1)
            max_val = np.max(arr, axis=1).reshape(arr.shape[0], 1)

        return np.round(arr / max_val * 100).astype(int)

    model = np.load('data.npy')

    one = resize_image(data)
    num = slice_number(one)

    diff = np.sum(np.abs(normalize(model[:, 1:]) - normalize(num)), axis=1)
    return np.array([model[np.argmin(diff), 0].astype(int), np.min(diff), np.round(100 - (np.min(diff) / np.max(diff) * 100)).astype(int)])
