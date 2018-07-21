'''
Adapted/taken from https://stackoverflow.com/questions/27704490/interactive-pixel-information-of-an-image-in-python

Thanks to Joe Kington

'''

import numpy as np
import matplotlib.pyplot as plt

class Formatter(object):
    def __init__(self, im):
        self.im = im
    def __call__(self, x, y):
        z = self.im.get_array()[int(y), int(x)]
        return 'x={:.01f}, y={:.01f}, z={:.01f}'.format(x, y, z)