import sys
import os
import json
import cv2 as cv


if len(sys.argv) < 3:
	print('Not enougth args: <folder> <type>')
	sys.exit()

folder = sys.argv[1]
code_type = int(sys.argv[2])
data = {'types_list': [{'id': 0, 'name': 'QR-code'}, {'id': 1, 'name': 'Data matrix'}], 'objects': []}
objects = []
float_precision = 4
display_scale = 0.23

for file in os.listdir(folder):
	file_path = os.path.join(folder, file)
	img = cv.imread(file_path)
	img = cv.resize(img, None, fx=display_scale, fy=display_scale)
	
	bboxes_tuple = cv.selectROIs("Select ROIs", img, False)
	markup = []

	for bbox_tuple in bboxes_tuple:
		bbox = [
			round(bbox_tuple[0] / img.shape[1], float_precision),
			round(bbox_tuple[1] / img.shape[0], float_precision),
			round(bbox_tuple[2] / img.shape[1], float_precision),
			round(bbox_tuple[3] / img.shape[0], float_precision)
		]
		markup.append({'type': code_type, 'bbox': bbox, 'bound': []})

	objects.append({'image': file, 'markup': markup})

data['objects'] = objects
with open(os.path.join(folder, 'markup.json'), 'w') as file:
	json.dump(data, file, indent=2)
