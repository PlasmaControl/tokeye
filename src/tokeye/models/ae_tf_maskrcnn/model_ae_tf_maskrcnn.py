from pathlib import Path

import torch
import torch.nn as nn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from torchvision.models.detection import (
    maskrcnn_resnet50_fpn_v2,
    MaskRCNN_ResNet50_FPN_V2_Weights,
)
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from .config_ae_tf_mask import AETFMaskConfig

class AETFMaskModel(nn.Module):
    def __init__(self, config: AETFMaskConfig):
        super().__init__()
        self.config = config

        model = maskrcnn_resnet50_fpn_v2(
            weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
            )
        in_features = model.roi_heads.box_predictor.cls_score.in_features  # type: ignore[union-attr]
        
        model.roi_heads.box_predictor = FastRCNNPredictor(
            in_features, 
            config.num_classes
            )
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels  # type: ignore[union-attr]
        
        model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask,
            config.hidden_layer,
            config.num_classes,
        )
        model.transform = GeneralizedRCNNTransform(
            min_size=config.min_size, # 800
            max_size=config.max_size, # 1333
            image_mean=config.image_mean,  # No normalization
            image_std=config.image_std,   # No normalization
        )

    def forward(self, input_BCHW: torch.Tensor) -> tuple[torch.Tensor]:
        return (logits,)