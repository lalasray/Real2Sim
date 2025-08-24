import numpy as np

npz_file = "/home/lala/Desktop/accelerometer.npy"
data = np.load(npz_file)

print(data.shape)