from PIL import Image
import numpy as np

class idxs():
    def __init__(self, dadafile, labelsfile):
        bindhead = np.fromfile(dadafile, dtype='>u4', count=4)
        self.bindata = np.fromfile(dadafile, dtype=np.ubyte)[16:]
        lblshead = np.fromfile(labelsfile, dtype='>u4', count=2)
        self.labels =  np.fromfile(labelsfile, dtype=np.ubyte)[8:]
        labelscount = lblshead[1]
        imagescount = bindhead[1]
        self.count = int(imagescount)
        assert labelscount==imagescount, "Count of label not equal count of images"
        self.rows = int(bindhead[2])
        self.cols = int(bindhead[3])
        self.binimages = self.bindata.reshape(imagescount, self.cols, self.rows)

    def getlabel(self, idx):
        return self.labels[idx]

    def getarray(self, idx):
        return self.binimages[idx]

    def getimage(self, idx):
        return Image.fromarray(self.binimages[idx])

#    def getbindata(self):
#        return self.bindata
#
#    def getlabels(self):
#        return self.labels
