import numpy as np
import cv2 as cv
import torch
import torchvision
import pytorch_lightning as pl
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


class FasterRCNN(pl.LightningModule):
    def __init__(self, classes):
        super().__init__()
        self.save_hyperparameters()
        self.classes = classes

        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
            weights='DEFAULT')
        self.in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(
            self.in_features, len(classes))
        self.model.train()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        return [optimizer]

    def training_step(self, batch, batch_idx):
        images, targets = batch

        self.model.train()
        loss_dict = self.model(images, targets)

        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses  # .item()

        # Logging to TensorBoard
        self.log("train_loss", loss_value)

        return loss_value

    def validation_step(self, batch, batch_idx):
        images, targets = batch

        loss_dict = self.model(images, targets)

        self.model.train()
        with torch.no_grad():
            loss_dict = self.model(images, targets)

        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses  # .item()

        # Logging to TensorBoard
        self.log("val_loss", loss_value)

        return loss_value


class Localizer_1:
    def __init__(self, model_checkpoint_path):
        self.model = FasterRCNN.load_from_checkpoint(model_checkpoint_path).model
        self.model.eval()

    def localize(self, img, thr=0.6):
        scale_size = 416
        scale = (img.shape[0] / scale_size, img.shape[1] / scale_size)
        input_img = cv.cvtColor(img, cv.COLOR_BGR2RGB).astype(np.float32)
        input_img_resized = cv.resize(input_img, (scale_size, scale_size))
        input_img_resized /= 255.0
        img = cv.resize(img, (scale_size, scale_size))

        results = self.model(torch.from_numpy(np.transpose(input_img_resized[None], (0, 3, 1, 2))))

        result = []

        for i in range(len(results[0]['labels'])):
            if results[0]['scores'][i] < thr:
                continue
                
            bbox = results[0]['boxes'][i]
            type = int(results[0]['labels'][i]) - 1
            x = int(bbox[0] * scale[1])
            y = int(bbox[1] * scale[0])
            w = int(bbox[2] * scale[1])
            h = int(bbox[3] * scale[0])
            h -= y
            w -= x
            
            result.append([x, y, w, h, type])

        return result
