from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights

class AETFMaskConfig:
    def __init__(
        self,
        num_classes: int = 2,
        hidden_layer: int = 256,
        min_size: int = 800,
        max_size: int = 1333,
        image_mean: list[float] = [0.0, 0.0, 0.0],
        image_std: list[float] = [1.0, 1.0, 1.0],
        weights: MaskRCNN_ResNet50_FPN_V2_Weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT,
        **kwargs,
    ):
        self.num_classes = num_classes
        self.hidden_layer = hidden_layer
        self.min_size = min_size
        self.max_size = max_size
        self.image_mean = image_mean
        self.image_std = image_std
        self.weights = weights