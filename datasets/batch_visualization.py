import os
import shutil
import random
import numpy as np
import torch
import cv2

from util.visualize import vis_output_seqs


class BatchVisualization:
    def __init__(self, epoch, max_length, root='bv', dir='bv'):
        self.dir = os.path.join(root, dir + '_' + str(epoch))
        self.epoch = epoch
        try:
            shutil.rmtree(self.dir)
        except:
            pass
        if not root in os.listdir():
            os.mkdir(root)
        os.mkdir(self.dir)
        self.max_length = max_length
        with open('data/dict.txt', 'r') as f:
            self.chars = f.readlines()[0]
    def bv(self, samples, ils):
        '''
        FIXME: Sometimes, there are wrong GTs in results
        '''
        gt_scores = torch.ones(1, 5000, 1000 + len(self.chars))
        imgs = vis_output_seqs(samples, ils, gt_scores, remove_padding=False, pad_rec=True, text_length=self.max_length, chars=self.chars)
        for img in imgs:
            name = os.path.join(
                self.dir,
                str(random.randrange(100_000, 999_999)) + '.jpg'
            )
            cv2.imwrite(name, img)