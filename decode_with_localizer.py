import sys
import os
import numpy as np
import json
import cv2 as cv
import pyzbar.pyzbar as pyzbar
from pylibdmtx import pylibdmtx
import pyzxing
from qr_dm_decoder.localizer_1 import Localizer_1
from qr_dm_decoder.localizer_2 import Localizer_2
import augmentation as aug

# QR
def detectAndDecodeOpenCV(img):
    bardet = cv.QRCodeDetector()

    try:
        decoded_info, corners, _ = bardet.detectAndDecode(img)
    except:
        print('OpenCV: Exception in QR decoder')
        return (False, '')

    if corners is None or len(decoded_info) == 0:
        print('OpenCV: QR-code not finded')
        return (False, '')

    return (True, decoded_info)

# QR
def detectAndDecodeZBar(img):
    decoded_objects = pyzbar.decode(img)
 
    if len(decoded_objects) < 1:
        print('PyZBar: Barcode not finded')
        return (False, '')

    obj = decoded_objects[0]

    return (True, obj.data.decode('utf-8'))

# DataMatrix
def detectAndDecodeLibDMtx(img):
    decoded_objects = pylibdmtx.decode(img)

    if len(decoded_objects) < 1:
        print('PyLibDMtx: Data matrix not finded')
        return (False, '')

    obj = decoded_objects[0]

    return (True, obj.data.decode('utf-8'))

# QR + DataMatrix
def detectAndDecodeZXing(img):
    bardet = pyzxing.BarCodeReader()
    decoded_objects = bardet.decode_array(img)
    
    obj = decoded_objects[0]

    if not 'points' in obj:
        print('PyZXing: Barcode not finded')
        return (False, '')

    return (True, obj['parsed'].decode('utf-8'))


if len(sys.argv) < 6:
    print('Usage: <markup file> <QR decoder> <DataMatrix decoder> <localizer> <localizer checkpoint> [augmentation]')
    sys.exit()

Augmentations = {
    'rotate': aug.rotate,
    'mix_channels': aug.mix_channels,
    'crop': aug.crop,
    'rotate_color': aug.rotate_color}

QR_decoders = {
    'opencv': detectAndDecodeOpenCV,
    'zbar': detectAndDecodeZBar,
    'zxing': detectAndDecodeZXing}

DataMatrix_decoders = {
    'libdmtx': detectAndDecodeLibDMtx,
    'zxing': detectAndDecodeZXing}

Localizers = {
    '1': Localizer_1,
    '2': Localizer_2}

markup_file = sys.argv[1]

QR_decoder_type = sys.argv[2]
DataMatrix_decoder_type = sys.argv[3]
Localizer_type = sys.argv[4]
localizer_checkpoint = sys.argv[5]

augmentation = None

if len(sys.argv) > 6:
    augmentation_type = sys.argv[6]

    if augmentation_type not in Augmentations:
        print('Unknown augmentation_type: ', augmentation_type)
        sys.exit()

    augmentation = Augmentations[augmentation_type]

if QR_decoder_type not in QR_decoders:
    print('Unknown QR_decoder_type: ', QR_decoder_type)
    sys.exit()

if DataMatrix_decoder_type not in DataMatrix_decoders:
    print('Unknown DataMatrix_decoder_type: ', DataMatrix_decoder_type)
    sys.exit()

if Localizer_type not in Localizers:
    print('Unknown Localizer_type: ', Localizer_type)
    sys.exit()

QR_decoder = QR_decoders[QR_decoder_type]
DataMatrix_decoder = DataMatrix_decoders[DataMatrix_decoder_type]
Localizer = Localizers[Localizer_type]

with open(markup_file, 'r') as file:
    data = json.load(file)

QR_decoded = 0
QR_total = 0
DataMatrix_decoded = 0
DataMatrix_total = 0
iter = 0
padding = 0.15
scale_size = int(130 * (1 + 2 * padding))
total = len(data['objects'])

localizer = Localizer(localizer_checkpoint)
aug.set_seed(0)

for object in data['objects']:
    img_path = os.path.join(os.path.dirname(markup_file), object['image'])
    img = cv.imread(img_path)

    bboxes = [m['bbox'] for m in object['markup']]

    x_key = lambda b: b[0]
    y_key = lambda b: b[1]
    x_all_min = min(bboxes, key=x_key)[0]
    y_all_min = min(bboxes, key=y_key)[1]
    x_all_max = max(bboxes, key=x_key)
    y_all_max = max(bboxes, key=y_key)
    all_bbox = [x_all_min,
                y_all_min,
                x_all_max[0] + x_all_max[2] - x_all_min - 1,
                y_all_max[1] + y_all_max[3] - y_all_min - 1]

    if augmentation is not None:
        img = augmentation(img, all_bbox)

    object['markup'] = []
    results = localizer.localize(img)

    for res in results:
        x = res[0]
        y = res[1]
        w = res[2]
        h = res[3]
        type = res[4]
        
        markup = {}
        markup['bbox'] = [x, y, w, h]
        markup['type'] = type

        orig_pad = int(padding * min(w, h))

        cropped_img = img[y - orig_pad : y + h + orig_pad, x - orig_pad : x + w + orig_pad]
        if cropped_img.shape[0] < 1 or cropped_img.shape[1] < 1:
            continue
        scale = scale_size / min(cropped_img.shape[0], cropped_img.shape[1])
        cropped_img = cv.resize(cropped_img, None, fx=scale, fy=scale, interpolation=cv.INTER_CUBIC)
        # cv.imshow('img', cropped_img)
        # cv.waitKey(0)

        if type == 0:
            success, info = QR_decoder(cropped_img)

            QR_total += 1

            if success:
                QR_decoded += 1
                
        elif type == 1:
            success, info = DataMatrix_decoder(cropped_img)

            DataMatrix_total += 1

            if success:
                DataMatrix_decoded += 1

        else:
            print('Unknown type: ', type)
            continue

        markup['decoded'] = success
        markup['decoded_info'] = info
        object['markup'].append(markup)

    iter += 1
    # Backup
    if iter % 5 == 0:
        result_file_path = os.path.join(os.path.dirname(markup_file), 'result.json')
        decoded_data = {'types_list': [{'id': 0, 'name': 'QR-code'}, {'id': 1, 'name': 'Data matrix'}]}
        decoded_data['objects'] = data['objects'][:iter]

        with open(result_file_path, 'w') as file:
            json.dump(decoded_data, file, indent=2)

        DataMatrix_percent = 0

        if DataMatrix_total > 0:
            DataMatrix_percent = round(DataMatrix_decoded / DataMatrix_total * 100, 1)

        QR_percent = 0

        if QR_total > 0:
            QR_percent = round(QR_decoded / QR_total * 100, 1)

        print(f'==================================\n {iter} of {total} images\n----------------------------------' +
            f'\nDecoder Type Decoded Total Percent\n' + 
            f'{QR_decoder_type:8} QR {QR_decoded:-6}{QR_total:-7}{QR_percent:-7}%\n' +
            f'{DataMatrix_decoder_type:8} DM {DataMatrix_decoded:-6}{DataMatrix_total:-7}{DataMatrix_percent:-7}%\n' + 
            '==================================')


result_file_path = os.path.join(os.path.dirname(markup_file), 'result.json')

with open(result_file_path, 'w') as file:
    json.dump(data, file, indent=2)

DataMatrix_percent = 0

if DataMatrix_total > 0:
    DataMatrix_percent = round(DataMatrix_decoded / DataMatrix_total * 100, 1)

QR_percent = 0

if QR_total > 0:
    QR_percent = round(QR_decoded / QR_total * 100, 1)

print(f'==================================\n Total {total} images\n----------------------------------' +
    f'\nDecoder Type Decoded Total Percent\n' + 
    f'{QR_decoder_type:8} QR {QR_decoded:-6}{QR_total:-7}{QR_percent:-7}%\n' +
    f'{DataMatrix_decoder_type:8} DM {DataMatrix_decoded:-6}{DataMatrix_total:-7}{DataMatrix_percent:-7}%\n' + 
    '==================================')
