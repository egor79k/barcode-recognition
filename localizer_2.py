from ultralytics import YOLO
import numpy as np
import torch
import torchvision
import math

class Localizer_2:
    def __init__(self, model):
        self.model = YOLO(model)

    def localize(self, img):
        results = self.model.predict(img)
        bboxes = (results[0].boxes.boxes).numpy()
        bboxes = bboxes.tolist()
        results = []
        for box in bboxes:
            del box[4]
            res = [math.ceil(number) for number in box]
            x = res[0]
            y = res[1]
            w = res[2] - x
            h = res[3] - y
            type = int(res[4])
            results.append([x, y, w, h, type])

        return results
