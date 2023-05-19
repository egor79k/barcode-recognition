import sys
import os
import numpy as np
import cv2 as cv
from qr_dm_decoder.decoder import decode


def displayResult(img, results):
    RED =   (127, 0, 255)
    GREEN = (0, 255, 0)
    BLUE =  (255, 255, 0)

    scale = 700 / img.shape[0]
    img = cv.resize(img, None, fx=scale, fy=scale)

    for code, polygon in results:
        polygon = (polygon * scale).astype(int)
        for pt in polygon:
            cv.circle(img, pt, 6, RED, 3)

        cv.polylines(img, [np.array([polygon], np.int32)], True, GREEN, 2)

        font = cv.FONT_HERSHEY_SIMPLEX
        cv.putText(img, code, polygon[3], font, 0.7, BLUE, 2)
        print('\nDECODED INFO:', code)

    if len(sys.argv) == 5:
        cv.imwrite(sys.argv[4], img)
    else:
        cv.imshow("BarCode", img)
        cv.waitKey(0);


if len(sys.argv) < 4:
    print('Usage: <image path> <localizer (1 or 2)> <localizer checkpoint> [result save path]')
    sys.exit()

img_path = sys.argv[1]
localizer_type = sys.argv[2]
localizer_checkpoint = sys.argv[3]

img = cv.imread(img_path)

decode_res = decode(img, localizer_type, localizer_checkpoint)

displayResult(img, decode_res)
