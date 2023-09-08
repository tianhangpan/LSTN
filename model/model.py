""" A plug and play Spatial Transformer Module in Pytorch """
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        # self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        # self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        # self.conv2_drop = nn.Dropout2d()
        # self.fc1 = nn.Linear(320, 50)
        # self.fc2 = nn.Linear(50, 10)

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7, padding=3),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5, padding=2),
            nn.AdaptiveAvgPool2d((3, 3))
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def forward(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 3 * 3)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size(), align_corners=False)
        x = F.grid_sample(x, grid)

        return x


class VGG16(nn.Module):
    def __init__(self, init_weights=True):
        super().__init__()
        front_features = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M']
        back_features = [512, 512, 512]

        self.front_end = self.make_layers(front_features, batch_norm=True)
        self.back_end = self.make_layers(back_features, in_channels=512, batch_norm=True)
        self.output_layer = nn.Conv2d(512, 1, 1)
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.front_end(x)
        x = self.back_end(x)
        x = self.output_layer(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    @staticmethod
    def make_layers(cfg, in_channels=3, batch_norm=False):
        layers = []
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)


class LSTN(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg = VGG16()
        self.stn = SpatialTransformer()

    def forward(self, x, stn_mode=True):
        m_reg = self.vgg(x)
        if stn_mode:
            m_lst = self.stn(m_reg)
            return m_reg, m_lst
        return m_reg

    def freeze_front_end(self):
        for params in self.vgg.front_end.parameters():
            params.requires_grad = False

    def unfreeze_front_end(self):
        for params in self.vgg.front_end.parameters():
            params.requires_grad = True


if __name__ == '__main__':
    model = LSTN()
    # a = torch.randn([4, 3, 128, 128])
    # b, c = model(a, stn_mode=True)
    # print(b.shape)
    # print(c.shape)
    model.freeze_front_end()
