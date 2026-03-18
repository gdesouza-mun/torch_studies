#https://deepnote.com/blog/ultimate-guide-to-torchvision-library-in-python
#First Tutorial from deepnote

import torch
import torchvision
from torchvision import transforms, models
from PIL import Image

# 1 Load an Image
image_path ="FudanPed00004.png"
image = Image.open(image_path) #PIL image object

# 2 Define
transform_pipeline = transforms.Compose(
    [transforms.Resize((224,224)), #ResNet18 expects 224x224 images
     transforms.ToTensor(), #Convert PIL to torch.Tensor
     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
     ]) #end of transform pipeline

# 3 Apply pipeline to image
tensor_image = transform_pipeline(image)
print("Transformed image tensor shape:", tensor_image.shape)
batch_tensor = tensor_image.unsqueeze(0)
print("Batch tensor shape:", batch_tensor.shape)

model = models.resnet18(pretrained=True) #Download weights


with torch.no_grad():
    model.eval() #set evaluation mode
    outputs = model(batch_tensor) #Contains logits for 1000 classes of ResNet

_, predicted_idx = outputs.max(dim=1) #Get highest score prediction
predict_idx = predicted_idx.item()
print("Predicted class index:", predicted_idx)

class_names = models.ResNet18_Weights.DEFAULT.meta["categories"]
predicted_label = class_names[predicted_idx]
print("Predicted label:", predicted_label)
