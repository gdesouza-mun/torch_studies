#https://docs.pytorch.org/tutorials/intermediate/torchvision_tutorial.html

import os
import torch
import matplotlib.pyplot as plt

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.io import read_image
from torchvision.ops.boxes import masks_to_boxes
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F
from torchvision.transforms import v2 as T
import utils



# image = read_image("PennFudanPed/PNGImages/FudanPed00046.png")
# mask = read_image("PennFudanPed/PedMasks/FudanPed00046_mask.png")

# #Instances are encoded as different colors
# print(mask.shape)
# obj_ids=torch.unique(mask)
# print(obj_ids)
# #First ID is the background, so remove it
# num_objs = len(obj_ids)

# masks = (mask == obj_ids[:, None, None]).to(dtype=torch.uint8)

# print(masks[-1])



# plt.figure(figsize=(16, 8))
# plt.subplot(121)
# plt.title("Image")
# plt.imshow(image.permute(1, 2, 0))
# plt.subplot(122)
# plt.title("Mask")
# plt.imshow(mask.permute(1, 2, 0))
# plt.show()

class PennFudanDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms):
        self.root=root
        self.transforms=transforms
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))


    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])

        img = read_image(img_path)
        mask = read_image(mask_path)

        #Instances are encoded as different colors
        obj_ids=torch.unique(mask)
        #First ID is the background, so remove it
        num_objs = len(obj_ids)

        masks = (mask == obj_ids[:, None, None]).to(dtype=torch.uint8)

        # Get bouding box coordinates for each mask
        boxes = masks_to_boxes(masks)

        # There is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)

        image_id = idx

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # Wrap sample and targets into torchvision tv_tensors:
        img = tv_tensors.Image(img)

        target = {}
        target["boxes"] = tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=F.get_size(img))
        target["masks"] = tv_tensors.Mask(masks)
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target


    def __len__(self):
        return len(self.imgs)



def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        hidden_layer,
        num_classes
    )

    return model


def get_transform(train):
    transforms = []
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms)



#== Forward Test


# model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
# dataset = PennFudanDataset('PennFudanPed', get_transform(train=True))
# data_loader = torch.utils.data.DataLoader(
#     dataset,
#     batch_size=2,
#     shuffle=True,
#     collate_fn=utils.collate_fn
# )

# # For Training
# images, targets = next(iter(data_loader))
# images = list(image for image in images)
# targets = [{k: v for k, v in t.items()} for t in targets]
# output = model(images, targets)  # Returns losses and detections
# print(output)

# # For inference
# model.eval()
# x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
# predictions = model(x)  # Returns predictions
# print(predictions[0])
