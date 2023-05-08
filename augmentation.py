import numpy as np
import cv2 as cv


def set_seed(seed):
    np.random.seed(seed)


def rotate(img, bbox):
    center = (img.shape[0] / 2, img.shape[1] / 2)
    angle = np.random.uniform(0, 360)
    matrix = cv.getRotationMatrix2D(center, angle, 1)
    return cv.warpAffine(img, matrix, img.shape[:2]), angle


def mix_channels(img, bbox):
    order = np.random.permutation(3)
    return img[:,:,order], [int(x) for x in order]


def crop(img, bbox):
    x = np.random.randint(0, bbox[0])
    y = np.random.randint(0, bbox[1])
    w = np.random.randint(bbox[0] + bbox[2], img.shape[1])
    h = np.random.randint(bbox[1] + bbox[3], img.shape[0])
    return img[y:h, x:w], [x, y, w - x, h - y]


def rotate_color(img, bbox):
    hue = np.random.uniform(0, 360)
    img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    img[:,:,0] = hue
    return cv.cvtColor(img, cv.COLOR_HSV2BGR), hue