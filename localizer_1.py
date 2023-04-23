import numpy as np
import cv2 as cv
import predict
import torch

class Localizer_1:
    def __init__(self, model_checkpoint_path):
        self.model = predict.FasterRCNN.load_from_checkpoint(model_checkpoint_path).model
        self.model.eval()

    def localize(self, img):
        input_img = cv.cvtColor(img, cv.COLOR_BGR2RGB).astype(np.float32)
        input_img_resized = cv.resize(input_img, (416, 416))
        input_img_resized /= 255.0
        img = cv.resize(img, (416, 416))

        results = self.model(torch.from_numpy(np.transpose(input_img_resized[None], (0, 3, 1, 2))))

        result = []

        for res in results:
            if len(res['boxes']) == 0:
                continue

            bbox = res['boxes'][0]
            type = int(res['labels'][0]) - 1
            x = int(bbox[0])
            y = int(bbox[1])
            w = int(bbox[2])
            h = int(bbox[3])
            h -= y
            w -= x

            result.append([x, y, w, h, type])

        return result