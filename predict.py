import pytorch_lightning as pl
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import numpy as np
import cv2

class FasterRCNN(pl.LightningModule):
    def __init__(self, classes):
        super().__init__()
        self.save_hyperparameters()
        self.classes = classes

        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        self.in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(self.in_features, len(classes))
        self.model.train()
    
    def configure_optimizers(self):
        #             ** MAYBE CHANGE OPTIMIZER **
        #optimizer = torch.optim.RMSprop(self.model.parameters(), lr=1e-4, eps=1e-8, weight_decay=5e-4, momentum=0.9)
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
        #lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5)
        return [optimizer]

    def training_step(self, batch, batch_idx):
        images, targets = batch

        #images = list(image.to('cuda') for image in images)
        #targets = [{k: v.to('cuda') for k, v in t.items()} for t in targets]

        #loss_dict = self.model(images, targets)
        #losses = sum(dict_['scores'].sum() for dict_ in loss_dict) #changed here
        #loss_value = losses.item()
        self.model.train()
        loss_dict = self.model(images, targets)

        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses#.item()

        # Logging to TensorBoard
        self.log("train_loss", loss_value)
        
        return loss_value
    
    def validation_step(self, batch, batch_idx):
        images, targets = batch

        #images = list(image.to('gpu') for image in images)
        #targets = [{k: v.to('gpu') for k, v in t.items()} for t in targets]
        
        loss_dict = self.model(images, targets)
        #print(len(loss_dict), loss_dict[0]['scores'].shape, loss_dict[1]['scores'].shape)
        #print(loss_dict)
        #losses = sum(dict_['scores'].sum() for dict_ in loss_dict)
        #loss_value = losses.item()
        self.model.train()
        with torch.no_grad():
            loss_dict = self.model(images, targets)

        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses#.item()
        
        # Logging to TensorBoard
        self.log("val_loss", loss_value)

        return loss_value


# model = None

# LOADING MODEL
def init_model():
    model_checkpoint_path = 'epoch=19-step=4540.ckpt' # PUT YOUR PATH HERE !
    model = FasterRCNN.load_from_checkpoint(model_checkpoint_path).model
    model.eval()

# GETTING PREDICTIONS
def localize(image):
    # image_path = "Barcode.v3i.voc/test/05102009115.rf.a2273c50fb8e7ac1bf9d0bf40e731de8.jpg" # PUT YOUR PATH HERE !
    # image = cv2.imread(image_path)

    width = 416
    height = 416
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    image_resized = cv2.resize(image, (416, 416))
    image_resized /= 255.0

    return model(torch.from_numpy(np.transpose(image_resized[None], (0, 3, 1, 2))))


# pred format:
#[{'boxes': tensor([[ 88.1090, 114.2962, 304.9707, 301.1288]], grad_fn=<StackBackward0>),
#  'labels': tensor([1]),
#  'scores': tensor([0.9986], grad_fn=<IndexBackward0>)}]
# pred is a list of dicts
#
# to draw one box:
# box = pred[0]['boxes'][i]
# cv2.rectangle(img_with_boxes, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color)
