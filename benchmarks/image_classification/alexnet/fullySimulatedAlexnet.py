import torch
import torch.nn as nn
from utils import load_state_dict_from_url

__all__ = ['FullySimulatedAlexNet', 'fullySimulatedalexnet']

model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}

class FullySimulatedAlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(FullySimulatedAlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.SimulatedConv2d(3, 64, kernel_size=11, path_to_arch_file='maeri-config/maeri_256mses_256_bw.cfg', path_to_tile='tiles/tile_configuration_conv1.txt', sparsity_ratio=0.90, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.SimulatedMaxPool2d(kernel_size=3, path_to_arch_file='maeri-config/pool_256mses_256_bw.cfg', path_to_tile='tiles/tile_configuration_pool.txt', stride=2),
            nn.SimulatedConv2d(64, 192, kernel_size=5,path_to_arch_file='maeri-config/maeri_256mses_256_bw.cfg', path_to_tile='tiles/tile_configuration_conv2.txt', sparsity_ratio=0.90, padding=2),
            nn.ReLU(inplace=True),
            nn.SimulatedMaxPool2d(kernel_size=3, path_to_arch_file='maeri-config/pool_256mses_256_bw.cfg', path_to_tile='tiles/tile_configuration_pool.txt', stride=2),
            nn.SimulatedConv2d(192, 384, kernel_size=3, path_to_arch_file='maeri-config/maeri_256mses_256_bw.cfg', path_to_tile='tiles/tile_configuration_conv3.txt', sparsity_ratio=0.90, padding=1),
            nn.ReLU(inplace=True),
            nn.SimulatedConv2d(384, 256, kernel_size=3, path_to_arch_file='maeri-config/maeri_256mses_256_bw.cfg', path_to_tile='tiles/tile_configuration_conv4.txt', sparsity_ratio=0.90,padding=1),
            nn.ReLU(inplace=True),
            nn.SimulatedConv2d(256, 256, kernel_size=3, path_to_arch_file='maeri-config/maeri_256mses_256_bw.cfg', path_to_tile='tiles/tile_configuration_conv5.txt', sparsity_ratio=0.90, padding=1),
            nn.ReLU(inplace=True),
            nn.SimulatedMaxPool2d(kernel_size=3, path_to_arch_file='maeri-config/pool_256mses_256_bw.cfg', path_to_tile='tiles/tile_configuration_pool.txt', stride=2),
        )

        self.avgpool = nn.SimulatedAdaptativeAvgPool2d((6, 6), path_to_arch_file='maeri-config/pool_256mses_256_bw.cfg', path_to_tile='tiles/tile_configuration_pool.txt')

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.SimulatedLinear(256 * 6 * 6, 4096, path_to_arch_file='maeri-config/maeri_256mses_256_bw.cfg', path_to_tile='tiles/tile_configuration_fc6.txt', sparsity_ratio=0.90),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.SimulatedLinear(4096, 4096,path_to_arch_file='maeri-config/maeri_256mses_256_bw.cfg', path_to_tile='tiles/tile_configuration_fc7.txt', sparsity_ratio=0.90),
            nn.ReLU(inplace=True),
            nn.SimulatedLinear(4096, num_classes, path_to_arch_file='maeri-config/maeri_256mses_256_bw.cfg', path_to_tile='tiles/tile_configuration_fc8.txt', sparsity_ratio=0.90),
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def fully_simulated_alexnet_model(pretrained=False, progress=True, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = FullySimulatedAlexNet(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['alexnet'],
                                           progress=progress)
        model.load_state_dict(state_dict)
    return model
