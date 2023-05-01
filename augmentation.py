import numpy as np
import cv2 as cv


def set_seed(seed):
    np.random.seed(seed)


def rotate(img):
    center = (img.shape[0] / 2, img.shape[1] / 2)
    angle = np.random.uniform(0, 360)
    matrix = cv.getRotationMatrix2D(center, angle, 1)
    return cv.warpAffine(img, matrix, img.shape[:2])


def mix_channels(img):
    order = np.random.permutation(3)
    return img[:,:,order]


def crop(img):
    x = np.random.randint(0, img.shape[1] / 2)
    y = np.random.randint(0, img.shape[0] / 2)
    w = np.random.randint(img.shape[1] / 2, img.shape[1])
    h = np.random.randint(img.shape[0] / 2, img.shape[0])
    return img[y:h, x:w]


def rotate_color(img):
    hue = np.random.uniform(0, 360)
    img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    img[:,:,0] = hue
    return cv.cvtColor(img, cv.COLOR_HSV2BGR)