import numpy as np
import fnmatch
import os
import engine


file_out_name = "data.npy"
path = "data_set"
output_arr = []

folders = os.listdir(path)

for folder in folders:
    images = fnmatch.filter(os.listdir(path + "/" + folder), '*.png')

    for image in images:
        roi = engine.prepare_image(path + "/" + folder + "/" + image)
        one = engine.resize_image(roi)
        num = engine.slice_number(one).astype(str)

        result = np.insert(num, 0, folder)
        output_arr = np.append(output_arr, result)

out = output_arr.reshape(-1, result.shape[0])
out = np.where(out == "dot", ".", out)
print("data:\n", out)
np.save(file_out_name, out)
