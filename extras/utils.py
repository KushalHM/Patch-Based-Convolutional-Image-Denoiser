import random
import cv2
import numpy as np


def addMultiLayerGaussian(image, maxCycles=15):
    noisy = image.copy()
    numCycles = random.randint(1,maxCycles)
    row,col,ch= image.shape
    for _ in range(numCycles):
        mean = random.random()
        var = random.random()*1000
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        noisy = noisy + gauss
    return noisy
