from ultralytics import YOLO
import numpy as np
import torch
import torchvision
import math

class Localizer:
    def __init__(self, model):
        self.model = YOLO(model)

    def localize(self, img):
        results = self.model.predict(img)
        bboxes = (results[0].boxes.boxes).numpy()
        bboxes = bboxes.tolist()
        round_bboxes = []
        for box in bboxes:
            del box[4]
            round_bboxes.append([math.ceil(number) for number in box])
        return round_bboxes