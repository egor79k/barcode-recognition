import sys
import os
import numpy as np
import json
import cv2 as cv
import pyzbar.pyzbar as pyzbar
from pylibdmtx import pylibdmtx
import pyzxing
import predict
import torch


# def displayResult(img, code, polygon):
#     RED =   (0, 0, 255)
#     GREEN = (0, 255, 0)
#     BLUE =  (255, 0, 0)

#     for pt in polygon:
#         cv.circle(img, pt, 6, RED, 3)

#     cv.polylines(img, [np.array([polygon], np.int32)], True, GREEN, 2)

#     font = cv.FONT_HERSHEY_SIMPLEX
#     cv.putText(img, code, (10, 30), font, 1.0, BLUE, 3)

#     cv.imshow("BarCode", img)
#     cv.waitKey(0);

# BarCode
def detectAndDecodeOpenCVBarcode(img):
    bardet = cv.barcode.BarcodeDetector()
    ok, decoded_info, decoded_type, corners = bardet.detectAndDecode(img)

    if not ok:
        print('OpenCV: Barcode not finded')
        return (False, '')

    # int_corners = tuple(tuple(map(int, x)) for x in corners[0])

    return (True, decoded_info[0])

# QR
def detectAndDecodeOpenCV(img):
    bardet = cv.QRCodeDetector()
    decoded_info, corners, _ = bardet.detectAndDecode(img)

    if corners is None:
        print('OpenCV: QR-code not finded')
        return (False, '')

    # int_corners = tuple(tuple(map(int, x)) for x in corners[0])

    # displayResult(img.copy(), decoded_info, int_corners)
    return (True, decoded_info)

# QR
def detectAndDecodeZBar(img):
    decoded_objects = pyzbar.decode(img)
 
    if len(decoded_objects) < 1:
        print('PyZBar: Barcode not finded')
        return (False, '')

    obj = decoded_objects[0]
    # int_corners = [[p.x, p.y] for p in obj.polygon]

    # displayResult(img.copy(), obj.data.decode('utf-8'), int_corners)
    return (True, obj.data.decode('utf-8'))

# DataMatrix
def detectAndDecodeLibDMtx(img):
    decoded_objects = pylibdmtx.decode(img)

    if len(decoded_objects) < 1:
        print('PyLibDMtx: Data matrix not finded')
        return (False, '')

    obj = decoded_objects[0]
    # r = obj.rect
    # int_corners = [[r.left, r.top], [r.left + r.width, r.top], [r.left + r.width, r.top + r.height], [r.left, r.top + r.height]]

    # displayResult(img.copy(), obj.data.decode('utf-8'), int_corners)
    return (True, obj.data.decode('utf-8'))

# QR + DataMatrix
def detectAndDecodeZXing(img):
    bardet = pyzxing.BarCodeReader()
    decoded_objects = bardet.decode_array(img)
    
    obj = decoded_objects[0]

    if not 'points' in obj:
        print('PyZXing: Barcode not finded')
        return (False, '')

    # int_corners = tuple((int(x[0]), int(x[1])) for x in obj['points'])

    # displayResult(img.copy(), obj['parsed'].decode('utf-8'), int_corners)
    return (True, obj['parsed'].decode('utf-8'))


if len(sys.argv) < 4:
    print('Usage: <markup file> <QR decoder> <DataMatrix decoder>')
    sys.exit()

QR_decoders = {
    'opencv': detectAndDecodeOpenCV,
    'zbar': detectAndDecodeZBar,
    'zxing': detectAndDecodeZXing}

DM_decoders = {
    'libdmtx': detectAndDecodeLibDMtx,
    'zxing': detectAndDecodeZXing}

dir_path = sys.argv[1]

QR_decoder_type = sys.argv[2]
DM_decoder_type = sys.argv[3]
BC_decoder_type = 'opencv'

if QR_decoder_type not in QR_decoders:
    print('Unknown QR_decoder_type: ', QR_decoder_type)
    sys.exit()

if DM_decoder_type not in DM_decoders:
    print('Unknown DM_decoder_type: ', DM_decoder_type)
    sys.exit()

QR_decoder = QR_decoders[QR_decoder_type]
DM_decoder = DM_decoders[DM_decoder_type]

# with open(markup_file, 'r') as file:
#     data = json.load(file)

BC_decoded = 0
BC_total = 0
QR_decoded = 0
QR_total = 0
DM_decoded = 0
DM_total = 0
iter = 0

objects = os.listdir(dir_path)
total = len(objects)

# predict.init_model()
model_checkpoint_path = 'epoch=19-step=4540.ckpt'
model = predict.FasterRCNN.load_from_checkpoint(model_checkpoint_path).model
model.eval()


# for object in data['objects']:
for img_name in objects:
    img_path = os.path.join(dir_path, img_name)
    img = cv.imread(img_path)

    # results = predict.localize(img)
    # width = 416
    # height = 416
    input_img = cv.cvtColor(img, cv.COLOR_BGR2RGB).astype(np.float32)
    input_img_resized = cv.resize(input_img, (416, 416))
    input_img_resized /= 255.0
    img_resized = cv.resize(img, (416, 416))

    results = model(torch.from_numpy(np.transpose(input_img_resized[None], (0, 3, 1, 2))))

    # for markup in object['markup']:
    for res in results:
        # bbox = markup['bbox']
        # type = markup['type']
        # img_height = img.shape[0]
        # img_width = img.shape[1]
        # x = int(bbox[0] * img_width)
        # y = int(bbox[1] * img_height)
        # w = int(bbox[2] * img_width)
        # h = int(bbox[3] * img_height)
        if len(res['boxes']) == 0:
            continue

        bbox = res['boxes'][0]
        type = res['labels'][0] + 1
        x = int(bbox[0])
        y = int(bbox[1])
        w = int(bbox[2])
        h = int(bbox[3])

        cropped_img = img_resized[y : h, x : w]

        # cv.normalize(img_resized, img_resized, 0, 1, cv.NORM_MINMAX)
        # img_with_boxes = img_resized.copy()
        # cv.rectangle(img_with_boxes, (x, y, w - x, h - y), (255, 0, 0), thickness=3)
        # cv.imshow("IMG", img_with_boxes)
        # cv.waitKey(0)
        # cv.imshow("IMG", cropped_img)
        # cv.waitKey(0)

        # cropped_img = img[y : y + h, x : x + w]
        # scale = 150 / min(cropped_img.shape[0], cropped_img.shape[1])
        # cropped_img = cv.resize(cropped_img, None, fx=scale, fy=scale)

        if cropped_img.shape[0] < 1 and cropped_img.shape[1] < 1:
            continue

        if type == 0:
            success, info = QR_decoder(cropped_img)

            QR_total += 1

            if success:
                QR_decoded += 1
                
        elif type == 1:
            success, info = DM_decoder(cropped_img)

            DM_total += 1

            if success:
                DM_decoded += 1

        elif type == 2:
            success, info = detectAndDecodeOpenCVBarcode(cropped_img)

            BC_total += 1

            if success:
                BC_decoded += 1

        else:
            print('Unknown type: ', type)
            continue

        # markup['decoded'] = success
        # markup['decoded_info'] = info

    iter += 1
    # Backup
    if iter % 5 == 0:
        # result_file_path = os.path.join(os.path.dirname(markup_file), 'result.json')
        # decoded_data = {'types_list': [{'id': 0, 'name': 'QR-code'}, {'id': 1, 'name': 'Data matrix'}]}
        # decoded_data['objects'] = data['objects'][:iter]

        # with open(result_file_path, 'w') as file:
        #     json.dump(decoded_data, file, indent=2)

        DM_percent = 0

        if DM_total > 0:
            DM_percent = round(DM_decoded / DM_total * 100, 1)

        QR_percent = 0

        if QR_total > 0:
            QR_percent = round(QR_decoded / QR_total * 100, 1)

        BC_percent = 0

        if BC_total > 0:
            BC_percent = round(BC_decoded / BC_total * 100, 1)

        print(f'==================================\n {iter} of {total} images\n----------------------------------' +
            f'\nDecoder Type Decoded Total Percent\n' + 
            f'{QR_decoder_type:8} QR {QR_decoded:-6}{QR_total:-7}{QR_percent:-7}%\n' +
            f'{DM_decoder_type:8} DM {DM_decoded:-6}{DM_total:-7}{DM_percent:-7}%\n' + 
            f'{BC_decoder_type:8} BC {BC_decoded:-6}{BC_total:-7}{BC_percent:-7}%\n' + 
            '==================================')


# result_file_path = os.path.join(os.path.dirname(markup_file), 'result.json')

# with open(result_file_path, 'w') as file:
#     json.dump(data, file, indent=2)

DM_percent = 0

if DM_total > 0:
    DM_percent = round(DM_decoded / DM_total * 100, 1)

QR_percent = 0

if QR_total > 0:
    QR_percent = round(QR_decoded / QR_total * 100, 1)

BC_percent = 0

if BC_total > 0:
    BC_percent = round(BC_decoded / BC_total * 100, 1)

print(f'==================================\n Total {total} images\n----------------------------------' +
    f'\nDecoder Type Decoded Total Percent\n' + 
    f'{QR_decoder_type:8} QR {QR_decoded:-6}{QR_total:-7}{QR_percent:-7}%\n' +
    f'{DM_decoder_type:8} DM {DM_decoded:-6}{DM_total:-7}{DM_percent:-7}%\n' +
    f'{BC_decoder_type:8} BC {BC_decoded:-6}{BC_total:-7}{BC_percent:-7}%\n' +
    '==================================')
