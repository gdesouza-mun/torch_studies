from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection


# Note: Running this example requires the COCO dataset to be available. # COCO dataset is large (~20GB for images + annotations), not downloaded by default. # For demonstration, assume 'coco/annotations/instances_val2017.json' and images in 'coco/val2017/' are present.

coco_root = "coco" # root directory containing 'val2017' images and 'annotations' folder
ann_file = f"{coco_root}/annotations/instances_val2017.json"
img_dir = f"{coco_root}/val2017"

coco_dataset = CocoDetection(root=img_dir, annFile=ann_file, transform=transforms.ToTensor())
print("Number of COCO validation images:", len(coco_dataset))

# Access one sample
image, target = coco_dataset[0]
print("Image size:", image.shape)              # e.g., torch.Size([3, 480, 640])
print("Number of objects in image:", len(target))
print("First object annotation:", target[0])
