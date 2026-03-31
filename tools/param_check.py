import sys
import torch
import torch.nn as nn
import timm
from src.models.advanced_models import create_advanced_model
from src.models.frequency_models import FrequencyAwareCNN

def count_params(model):
    return sum(p.numel() for p in model.parameters())

print(f"EffNetB0: {count_params(timm.create_model('efficientnet_b0', pretrained=False))/1e6:.2f}M", flush=True)
print(f"Xception: {count_params(timm.create_model('xception', pretrained=False))/1e6:.2f}M", flush=True)
print(f"ResNet50: {count_params(timm.create_model('resnet50', pretrained=False))/1e6:.2f}M", flush=True)

model_h = create_advanced_model("hybrid_cnn_transformer", pretrained=False)
print(f"Hybrid: {count_params(model_h)/1e6:.2f}M", flush=True)

model_m = create_advanced_model("multiscale", pretrained=False)
print(f"MultiScale: {count_params(model_m)/1e6:.2f}M", flush=True)

model_f = FrequencyAwareCNN(pretrained=False)
print(f"FrequencyAware: {count_params(model_f)/1e6:.2f}M", flush=True)
