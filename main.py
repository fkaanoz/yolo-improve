from ultralytics import YOLO
import torch

# build model from yaml
model = YOLO("ultralytics/cfg/models/11/yolo11-fkz.yaml")
model.info()  # prints layers, params, GFLOPs — should be ~2M params

# dummy forward pass
x = torch.randn(1, 3, 640, 640)
y = model.model(x)
print("forward pass OK")