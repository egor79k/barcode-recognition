import sys
import os
import numpy as np
import cv2 as cv
from pylibdmtx import pylibdmtx
from localizer_1 import Localizer_1
from localizer_2 import Localizer_2


padding = 20
scale_size = 130 + padding * 2


def displayResult(img, results):
    RED =   (0, 0, 255)
    GREEN = (0, 255, 0)
    BLUE =  (255, 0, 0)

    scale = 700 / img.shape[0]
    img = cv.resize(img, None, fx=scale, fy=scale)

    for code, polygon in results:
        polygon = (polygon * scale).astype(int)
        for pt in polygon:
            cv.circle(img, pt, 6, RED, 3)

        cv.polylines(img, [np.array([polygon], np.int32)], True, GREEN, 2)

        font = cv.FONT_HERSHEY_SIMPLEX
        cv.putText(img, code, polygon[3], font, 0.5, BLUE, 2)
        print('\nDECODED INFO:', code)

    if len(sys.argv) == 5:
        cv.imwrite(sys.argv[4], img)
    else:
        cv.imshow("BarCode", img)
        cv.waitKey(0);


def detectAndDecodeOpenCV(img):
    bardet = cv.QRCodeDetector()

    try:
        decoded_info, corners, _ = bardet.detectAndDecode(img)
    except:
        print('OpenCV: Exception in QR decoder')
        return (False, '', ())

    if corners is None:
        print('OpenCV: QR-code not finded')
        return (False, '', ())

    int_corners = tuple(tuple(map(int, x)) for x in corners[0])

    return (True, decoded_info, int_corners)


def detectAndDecodeLibDMtx(img):
    decoded_objects = pylibdmtx.decode(img)

    if len(decoded_objects) < 1:
        print('PyLibDMtx: Data matrix not finded')
        return (False, '', ())

    obj = decoded_objects[0]
    r = obj.rect
    int_corners = [[r.left, r.top], [r.left + r.width, r.top], [r.left + r.width, r.top + r.height], [r.left, r.top + r.height]]

    return (True, obj.data.decode('utf-8'), int_corners)


if len(sys.argv) < 4:
    print('Usage: <image path> <localizer (1 or 2)> <localizer checkpoint> [result save path]')
    sys.exit()

Localizers = {
    '1': Localizer_1,
    '2': Localizer_2}

img_path = sys.argv[1]
Localizer_type = sys.argv[2]
localizer_checkpoint = sys.argv[3]

if Localizer_type not in Localizers:
    print('Unknown Localizer_type: ', Localizer_type)
    sys.exit()

Localizer = Localizers[Localizer_type]
localizer = Localizer(localizer_checkpoint)

img = cv.imread(img_path)
results = localizer.localize(img)
decode_res = []

for res in results:
    x = res[0]
    y = res[1]
    w = res[2]
    h = res[3]
    type = res[4]

    cropped_img = img[y - padding : y + h + padding, x - padding : x + w + padding]
    if cropped_img.shape[0] < 1 or cropped_img.shape[1] < 1:
        continue
    scale = scale_size / min(cropped_img.shape[0], cropped_img.shape[1])
    cropped_img = cv.resize(cropped_img, None, fx=scale, fy=scale, interpolation=cv.INTER_CUBIC)

    if type == 0:
        success, info, corners = detectAndDecodeOpenCV(cropped_img)
    elif type == 1:
        success, info, corners = detectAndDecodeLibDMtx(cropped_img)
    else:
        print('Unknown type: ', type)
        continue

    if success:
        corners = (np.array(corners) / scale).astype(int)
        corners[:,1] += (y - padding)
        corners[:,0] += (x - padding)
        decode_res.append([info, corners])

displayResult(img, decode_res)
