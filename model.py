import models

class Model:
    def __init__(
        self,
        encoder_name="resnet50",
        decoder_name="unet",
        classes=3,
        pooling="avg",
        dropout=0.5,
        encoder_weights=None,
        in_channels=3,
        unet_class=2,
    ):
        self.encoder_name = encoder_name
        self.decoder_name = decoder_name
        aux_params = dict(
            pooling=pooling,  # one of 'avg', 'max'
            dropout=dropout,  # dropout ratio, default is None
            classes=classes,  # define number of output labels
        )
        self.model = models.Unet(
            encoder_name=encoder_name,  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights=encoder_weights,  # use `imagenet` pre-trained weights for encoder initialization
            in_channels=in_channels,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=unet_class,  # Binary output for unet
            aux_params=aux_params,  # model output channels (number of classes in your dataset)
        )

    def get_model(self):
        return self.model
