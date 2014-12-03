from PIL import Image
import numpy as np
from random import randint

bindhead = np.fromfile("train-images-idx3-ubyte.idx", dtype='>u4', count=4)
bindata = np.fromfile("train-images-idx3-ubyte.idx", dtype=np.ubyte)[16:]
lblshead = np.fromfile("train-labels-idx1-ubyte.idx", dtype='>u4', count=2)
labels =  np.fromfile("train-labels-idx1-ubyte.idx", dtype=np.ubyte)[8:]
labelscount = lblshead[1]
imagescount = bindhead[1]
rows = bindhead[2]
cols = bindhead[3]
print("rows = {0}, cols = {1}, labelscount={2}, imagescount={3}".format(rows, cols, labelscount, imagescount))

sizex = 28
sizey = 28
count = 60000
binimages = bindata.reshape(count, sizex, sizey)
idx = randint(0, count)
im = Image.fromarray(binimages[idx], mode="L")
print(labels[idx])
im.show()
