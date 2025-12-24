from pathlib import Path

import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from torchvision.models.detection import (
    maskrcnn_resnet50_fpn_v2,
    MaskRCNN_ResNet50_FPN_V2_Weights,
)
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

def build_model(num_classes=2, hidden_layer=256):
    weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = maskrcnn_resnet50_fpn_v2(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features  # type: ignore[union-attr]
    
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels  # type: ignore[union-attr]
    
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        hidden_layer,
        num_classes,
    )
    model.transform = GeneralizedRCNNTransform(
        min_size=800,
        max_size=1333,
        image_mean=[0.0, 0.0, 0.0],  # No normalization
        image_std=[1.0, 1.0, 1.0],   # No normalization
    )
    return model


def load_model(model_path, *args, **kwargs):
    # Create model
    model = build_model(*args, **kwargs)
    
    # Load weights
    model.load_state_dict(torch.load(
        model_path, 
        map_location=device,
        weights_only=True,
    ))
    return model