import torch
from contextlib import contextmanager, nullcontext
from torch import nn
from torch.utils.checkpoint import checkpoint
from torchvision.models import resnet18, resnet50


@contextmanager
def isolated_bn_recompute(modules):
    """Recompute with disposable BN buffers, retaining exactly one real update."""
    saved = []
    try:
        for module in modules:
            for layer in module.modules():
                if isinstance(layer, nn.modules.batchnorm._BatchNorm) and layer.track_running_stats:
                    buffers = (layer.running_mean, layer.running_var, layer.num_batches_tracked)
                    saved.append((layer, buffers))
                    layer.running_mean, layer.running_var, layer.num_batches_tracked = [v.clone() for v in buffers]
        yield
    finally:
        for layer, buffers in saved:
            layer.running_mean, layer.running_var, layer.num_batches_tracked = buffers


class ForceRegressor(nn.Module):
    """Only image pixels enter the model. No force history or deformation maps."""

    def __init__(self, output_dim=3, backbone='resnet18'):
        super().__init__()
        network = {'resnet18': resnet18, 'resnet50': resnet50}[backbone](weights=None)
        features = network.fc.in_features
        network.conv1 = nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False)
        self.encoder = nn.Sequential(*list(network.children())[:-2])
        self.activation_checkpointing = False
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 5)), nn.Flatten(),
            nn.Linear(features * 4 * 5, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, output_dim),
        )

    def forward(self, image):
        if image.ndim != 4 or image.shape[1:] != (1, 240, 320):
            raise ValueError(f"Expected B,1,240,320, got {tuple(image.shape)}")
        if not (self.activation_checkpointing and self.training and torch.is_grad_enabled()):
            return self.head(self.encoder(image))
        # Stem and each complete residual stage are independently recomputed.
        groups = [list(self.encoder[:4]), *[[stage] for stage in self.encoder[4:]]]
        x = image
        for modules in groups:
            def run(value, modules=modules):
                for module in modules:
                    value = module(value)
                return value
            def contexts(modules=modules):
                return nullcontext(), isolated_bn_recompute(modules)
            x = checkpoint(run, x, use_reentrant=False, context_fn=contexts)
        return self.head(x)
