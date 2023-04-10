import sys
import os
import numpy as np
import json
import cv2 as cv
import pyzbar.pyzbar as pyzbar
from pylibdmtx import pylibdmtx
import pyzxing
from localizer import Localizer

# QR
def detectAndDecodeOpenCV(img):
    bardet = cv.QRCodeDetector()
    decoded_info, corners, _ = bardet.detectAndDecode(img)

    if corners is None:
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


if len(sys.argv) < 4:
    print('Usage: <markup file> <QR decoder> <DataMatrix decoder>')
    sys.exit()

QR_decoders = {
    'opencv': detectAndDecodeOpenCV,
    'zbar': detectAndDecodeZBar,
    'zxing': detectAndDecodeZXing}

DataMatrix_decoders = {
    'libdmtx': detectAndDecodeLibDMtx,
    'zxing': detectAndDecodeZXing}

markup_file = sys.argv[1]

QR_decoder_type = sys.argv[2]
DataMatrix_decoder_type = sys.argv[3]

if QR_decoder_type not in QR_decoders:
    print('Unknown QR_decoder_type: ', QR_decoder_type)
    sys.exit()

if DataMatrix_decoder_type not in DataMatrix_decoders:
    print('Unknown DataMatrix_decoder_type: ', DataMatrix_decoder_type)
    sys.exit()

QR_decoder = QR_decoders[QR_decoder_type]
DataMatrix_decoder = DataMatrix_decoders[DataMatrix_decoder_type]

with open(markup_file, 'r') as file:
    data = json.load(file)

QR_decoded = 0
QR_total = 0
DataMatrix_decoded = 0
DataMatrix_total = 0
iter = 0
total = len(data['objects'])

localizer = Localizer('best (4).pt')

for object in data['objects']:
    img_path = os.path.join(os.path.dirname(markup_file), object['image'])
    img = cv.imread(img_path)
    object['markup'] = []
    bboxes = localizer.localize(img)

    for bbox in bboxes:
        x = bbox[0]
        y = bbox[1]
        w = bbox[2] - x
        h = bbox[3] - y
        
        type = int(bbox[4])
        
        markup = {}
        markup['bbox'] = [x, y, w, h]
        markup['type'] = type

        cropped_img = img[y : y + h, x : x + w]
        scale = 190 / min(cropped_img.shape[0], cropped_img.shape[1])
        cropped_img = cv.resize(cropped_img, None, fx=scale, fy=scale)

        if cropped_img.shape[0] < 1 and cropped_img.shape[1] < 1:
            continue

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
