import os
import shutil
import random
import numpy as np
import cv2


class BatchVisualization:
    def __init__(self, epoch, root='bv', dir='bv'):
        self.dir = os.path.join(root, dir + '_' + str(epoch))
        self.epoch = epoch
        try:
            shutil.rmtree(self.dir)
        except:
            pass
        if not root in os.listdir():
            os.mkdir(root)
        os.mkdir(self.dir)
    def bv(self, samples): #, ibs, ils, obs, ols):
        # TODO: Visualization of Ground Truth
        for sample in samples:
            img = sample.clone().numpy()
            img = np.transpose(img, (1,2,0))
            img *= 255
            img = img.astype(np.uint8)
            name = os.path.join(
                self.dir,
                str(random.randrange(100_000, 999_999)) + '.jpg'
            )
            cv2.imwrite(name, img)