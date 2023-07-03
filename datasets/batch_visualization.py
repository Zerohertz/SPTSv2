import os
import shutil
import random
import numpy as np
import cv2


def bv(samples): #, ibs, ils, obs, ols):
    # TODO: Visualization of Ground Truth
    dir = 'AugRes'
    try:
        shutil.rmtree(dir)
    except:
        pass
    os.mkdir(dir)
    for sample in samples:
        img = sample.clone().numpy()
        img = np.transpose(img, (1,2,0))
        IMAGENET_MEAN, IMAGENET_STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
        img = np.clip(255.0 * (img * IMAGENET_STD + IMAGENET_MEAN), 0, 255)
        name = dir + '/' + str(random.randrange(100_000, 999_999)) + '.jpg'
        cv2.imwrite(name, img)