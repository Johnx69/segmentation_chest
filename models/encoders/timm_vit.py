from timm.models.vision_transformer import VisionTransformer, vit_base_patch16_224_in21k
import torch.nn as nn
import torch
from ._base import EncoderMixin
import numpy as np


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(MLP, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.fc(x)


class VisionTransformerEncoder(VisionTransformer, EncoderMixin):
    def __init__(self, out_channels, depth=5, **kwargs):
        super().__init__(**kwargs)
        self._out_channels = out_channels
        self._depth = depth
        self._in_channels = 3

        # Remove final classification head
        self.head = nn.Identity()

    def get_stages(self):
        block_outputs = [
            torch.Tensor(self.patch_embed),
            torch.Tensor(self.blocks[:2]),
            torch.Tensor(self.blocks[2:4]),
            torch.Tensor(self.blocks[4:6]),
            torch.Tensor(self.blocks[6:]),
        ]

        input_dims = [output.shape[1:] for output in block_outputs]
        output_dims = [
            (64, 128, 128),
            (256, 64, 64),
            (512, 32, 32),
            (1024, 16, 16),
            (2048, 8, 8),
        ]

        # Initialize the MLPs
        mlps = [
            MLP(np.prod(input_dim), np.prod(output_dim))
            for input_dim, output_dim in zip(input_dims, output_dims)
        ]

        # Convert the block outputs to torch tensors
        input_tensors = [output.view(output.size(0), -1) for output in block_outputs]

        # Apply the MLPs
        transformed_tensors = [
            mlp(input_tensor) for mlp, input_tensor in zip(mlps, input_tensors)
        ]

        # Convert the output tensors back to cuda tensors and reshape to the desired size
        transformed_tensors_cuda = [
            output_tensor.view(output_tensor.size(0), *output_dim).to("cuda")
            for output_tensor, output_dim in zip(transformed_tensors, output_dims)
        ]

        return [nn.Identity(), *transformed_tensors_cuda]

    def forward(self, x):
        stages = self.get_stages()

        features = []
        for i in range(self._depth + 1):
            x = stages[i](x)
            features.append(x)

        return features

    def load_state_dict(self, state_dict, **kwargs):
        state_dict.pop("head.weight", None)
        state_dict.pop("head.bias", None)
        super().load_state_dict(state_dict, **kwargs)


timm_vit_encoders = {
    "timm-vit-base-patch16-224": {
        "encoder": VisionTransformerEncoder,
        "pretrained_settings": {
            "imagenet21k": {
                "mean": [0.5, 0.5, 0.5],
                "std": [0.5, 0.5, 0.5],
                "url": vit_base_patch16_224_in21k,
                "input_space": "RGB",
                "input_range": [0, 1],
            },
        },
        "params": {
            "out_channels": (3, 768, 768, 768, 768, 768, 768),
            "depth": 5,
            "img_size": 256,
            "patch_size": 16,
            "in_chans": 3,
            "num_classes": 21843,  # This will not be used, but needs to be specified
            "embed_dim": 768,
            "num_heads": 12,
            "mlp_ratio": 4,
            "qkv_bias": True,
            "drop_rate": 0.0,
            "drop_path_rate": 0.1,
            "norm_layer": nn.LayerNorm,
        },
    },
}
