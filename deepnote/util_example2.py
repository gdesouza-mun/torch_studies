from torchvision.utils import draw_bounding_boxes
import torchvision
import matplotlib.pyplot as plt

image = torchvision.io.read_image("FudanPed00004.png")
boxes = outputs[0]["boxes"]
labels = outputs[0]["labels"]
scores = outputs[0]["scores"]

#Only keep scores we are confident with
keep = scores > 0.5
boxes_keep = boxes[keep]
labels_keep = labels[keep]
scores_keep = scores[keep]
