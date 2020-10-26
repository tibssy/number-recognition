import numpy as np
import engine


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

    one = engine.resize_image(data)
    num = engine.slice_number(one)

    diff = np.sum(np.abs(normalize(model[:, 1:]) - normalize(num)), axis=1)
    return np.array([model[np.argmin(diff), 0].astype(int), np.min(diff), np.round(100 - (np.min(diff) / np.max(diff) * 100)).astype(int)])
