class BigTFUNetConfig:

    model_type = "big_tf_unet"

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 2,
        num_layers: int = 5,
        first_layer_size: int = 32,
        dropout_rate: float = 0.2,
        **kwargs,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.first_layer_size = first_layer_size
        self.dropout_rate = dropout_rate