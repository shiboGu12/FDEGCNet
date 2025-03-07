#!/Library/Frameworks/Python.framework/Versions/3.5/bin/python3.5
import math
import torch.nn as nn
import torch
from .GDN import GDN
from .ODAM import ODAM
class Synthesis_transform(nn.Module):

    def __init__(self, num_filters=128):
        super(Synthesis_transform, self).__init__()
        # i = 0
        self.attention1 = ODAM(num_filters)
        self.b0_layer0 = nn.ConvTranspose2d(num_filters, num_filters, 3, stride=1, padding=1)
        self.b0_layer0_relu = nn.LeakyReLU()
        self.b0_layer1 = nn.ConvTranspose2d(num_filters, num_filters, 3, stride=1, padding=1)
        self.b0_layer1_relu = nn.LeakyReLU()
        self.b0_shortcut = nn.ConvTranspose2d(num_filters, num_filters, 1, stride=2, output_padding=1)
        self.b0_layer2 = nn.ConvTranspose2d(num_filters, num_filters, 3, stride=2, padding=1, output_padding=1)
        self.b0_layer2_relu = nn.LeakyReLU()
        self.b0_layer3 = nn.ConvTranspose2d(num_filters, num_filters, 3, stride=1, padding=1)
        self.b0_layer3_igdn = GDN(num_filters, inverse=True)

        # i = 1
        self.b1_layer0 = nn.ConvTranspose2d(num_filters, num_filters, 3, stride=1, padding=1)
        self.b1_layer0_relu = nn.LeakyReLU()
        self.b1_layer1 = nn.ConvTranspose2d(num_filters, num_filters, 3, stride=1, padding=1)
        self.b1_layer1_relu = nn.LeakyReLU()
        self.b1_shortcut = nn.ConvTranspose2d(num_filters, num_filters, 1, stride=2, output_padding=1)
        self.b1_layer2 = nn.ConvTranspose2d(num_filters, num_filters, 3, stride=2, padding=1, output_padding=1)
        self.b1_layer2_relu = nn.LeakyReLU()
        self.b1_layer3 = nn.ConvTranspose2d(num_filters, num_filters, 3, stride=1, padding=1)
        self.b1_layer3_igdn = GDN(num_filters, inverse=True)

        # i = 2
        self.attention2 = ODAM(num_filters)
        self.b2_layer0 = nn.ConvTranspose2d(num_filters, num_filters, 3, stride=1, padding=1)
        self.b2_layer0_relu = nn.LeakyReLU()
        self.b2_layer1 = nn.ConvTranspose2d(num_filters, num_filters, 3, stride=1, padding=1)
        self.b2_layer1_relu = nn.LeakyReLU()
        self.b2_shortcut = nn.ConvTranspose2d(num_filters, num_filters, 1, stride=2, output_padding=1)
        self.b2_layer2 = nn.ConvTranspose2d(num_filters, num_filters, 3, stride=2, padding=1, output_padding=1)
        self.b2_layer2_relu = nn.LeakyReLU()
        self.b2_layer3 = nn.ConvTranspose2d(num_filters, num_filters, 3, stride=1, padding=1)
        self.b2_layer3_igdn = GDN(num_filters, inverse=True)

        # i = 3
        self.b3_layer0 = nn.ConvTranspose2d(num_filters, num_filters, 3, stride=1, padding=1)
        self.b3_layer0_relu = nn.LeakyReLU()
        self.b3_layer1 = nn.ConvTranspose2d(num_filters, num_filters, 3, stride=1, padding=1)
        self.b3_layer1_relu = nn.LeakyReLU()
        self.b3_layer2 = nn.ConvTranspose2d(num_filters, 3, 3, stride=2, padding=1, output_padding=1)



    def forward(self, x):
        # i = 0
        attention0 = self.attention1(x)
        b0 = self.b0_layer0_relu(self.b0_layer0(attention0))
        b0 = self.b0_layer1_relu(self.b0_layer1(b0))
        b0 += attention0
        shortcut0 = self.b0_shortcut(b0)
        b0 = self.b0_layer2_relu(self.b0_layer2(b0))
        b0 = self.b0_layer3_igdn(self.b0_layer3(b0))
        b0 += shortcut0

        # i = 1
        b1 = self.b1_layer0_relu(self.b1_layer0(b0))
        b1 = self.b1_layer1_relu(self.b1_layer1(b1))
        b1 += b0
        shortcut1 = self.b1_shortcut(b1)
        b1 = self.b1_layer2_relu(self.b1_layer2(b1))
        b1 = self.b1_layer3_igdn(self.b1_layer3(b1))
        b1 += shortcut1

        # i = 2
        attention2 = self.attention2(b1)
        b2 = self.b2_layer0_relu(self.b2_layer0(attention2))
        b2 = self.b2_layer1_relu(self.b2_layer1(b2))
        b2 += attention2
        shortcut2 = self.b2_shortcut(b2)
        b2 = self.b2_layer2_relu(self.b2_layer2(b2))
        b2 = self.b2_layer3_igdn(self.b2_layer3(b2))
        b2 += shortcut2

        # i = 3
        b3 = self.b3_layer0_relu(self.b3_layer0(b2))
        b3 = self.b3_layer1_relu(self.b3_layer1(b3))
        b3 += b2
        b3 = self.b3_layer2(b3)

        return b3

# if __name__ == "__main__":
#     # analysis_transform = Analysis_transform()
#     # analysis_hyper = Hyper_analysis()
#     # synthesis_hyper = Hyper_synthesis()
#     Synthesis_transform = Synthesis_transform()
#     # # input_image = torch.zeros([1, 128, 32, 48])
#     # input_image = torch.zeros([1, 3, 512, 768])
#     # y = analysis_transform(input_image)
#     # z = analysis_hyper(y)
#     # x_ = synthesis_transform(y)
#     # sigma = synthesis_hyper(z)
#     # print("y: ", y.shape)
#     # print("z: ", z.shape)
#     # print("x_: ", x_.shape)
#     # print("sigma: ", sigma.shape)
#     feature = torch.zeros([1,128,32,48])
#     output_image = Synthesis_transform(feature)
#     print(output_image.shape)
