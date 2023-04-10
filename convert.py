import sys
import os
import json
import cv2 as cv


if len(sys.argv) < 2:
    print('Usage: <dataset_folder>')
    sys.exit()

folder = sys.argv[1]
data = {'types_list': [{'id': 0, 'name': 'QR-code'}, {'id': 1, 'name': 'Data matrix'}], 'objects': []}
objects = []
# float_precision = 4

imgs_folder = os.path.join(folder, 'images')
markup_folder = os.path.join(folder, 'labels')
img_files = sorted(os.listdir(imgs_folder))
markup_files = sorted(os.listdir(markup_folder))

for img_file, markup_file in zip(img_files, markup_files):
    markup_path = os.path.join(markup_folder, markup_file)
    img_path = os.path.join(imgs_folder, img_file)
    img = cv.imread(img_path)
    markup = []

    with open(markup_path) as f:
        for line in f.read().split('\n'):
            if len(line) < 1:
                continue
            yolo_markup = line.split()
            code_type = -(int(yolo_markup[0]) - 1)
            bbox = [float(x) for x in yolo_markup[1:]]
            
            img_h = img.shape[0]
            img_w = img.shape[1]
            
            bbox[0] = bbox[0] - bbox[2] / 2
            bbox[1] = bbox[1] - bbox[3] / 2
            
            bbox[0] *= img_w
            bbox[1] *= img_h
            bbox[2] *= img_w
            bbox[3] *= img_h
            
            markup.append({'type': code_type, 'bbox': bbox, 'bound': []})
            # cv.imshow('IMG', img[int(bbox[1]) : int(bbox[1] + bbox[3]), int(bbox[0]):int(bbox[0] + bbox[2])])
            # cv.waitKey(0)

    objects.append({'image': img_file, 'markup': markup})

data['objects'] = objects
with open(os.path.join(folder, 'markup.json'), 'w') as file:
    json.dump(data, file, indent=2)
