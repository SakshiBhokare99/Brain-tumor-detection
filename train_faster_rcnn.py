import torch
from models.faster_rcnn import get_model

device = "cuda" if torch.cuda.is_available() else "cpu"

model = get_model().to(device)

print("Faster R-CNN Model Loaded")
print("Ready for training with bounding box dataset.")
torch.save(model.state_dict(),"saved_models/fasterrcnn.pth")