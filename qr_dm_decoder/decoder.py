import numpy as np
import cv2 as cv
from pylibdmtx import pylibdmtx
from .localizer_1 import Localizer_1
from .localizer_2 import Localizer_2


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

    if len(decoded_info) == 0:
        print('OpenCV: QR-code not decoded')
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


def decode(img, localizer_type : str, localizer_checkpoint : str):
    Localizers = {
        '1': Localizer_1,
        '2': Localizer_2}

    if localizer_type not in Localizers:
        print('Unknown localizer_type: ', localizer_type)
        return []

    Localizer = Localizers[localizer_type]
    localizer = Localizer(localizer_checkpoint)

    results = localizer.localize(img)
    decode_res = []

    padding = 0.15
    scale_size = int(130 * (1 + 2 * padding))

    for res in results:
        x = res[0]
        y = res[1]
        w = res[2]
        h = res[3]
        type = res[4]

        orig_pad = int(padding * min(w, h))

        cropped_img = img[y - orig_pad : y + h + orig_pad, x - orig_pad : x + w + orig_pad]
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
            corners[:,1] += (y - orig_pad)
            corners[:,0] += (x - orig_pad)
            decode_res.append([info, corners])

    return decode_res
