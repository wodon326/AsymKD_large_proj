

import torch
import torch.nn as nn
import timm
import torch.nn.functional as F

    

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)
    
    
class Double_DepthWise_Conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, groups=in_channels),
            nn.Conv2d(out_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, groups=out_channels),
            nn.Conv2d(out_channels, out_channels, 1),
        )
    
    def forward(self, x):
        return self.conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.pool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class ResidualBlock(nn.Module):
    """
    DepthwiseSeparableConv 블록 두 개를 쌓은 뒤,
    입력 x를 skip connection으로 더해주는 Residual Block
    """
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = Double_DepthWise_Conv(channels, channels)
        self.conv2 = Double_DepthWise_Conv(channels, channels)

    def forward(self, x):
        # skip connection
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out += identity  # Residual 연결
        return out

class Residual_Blocks(nn.Module):
    """
    channels=384인 입력에 대해 여러 개의 ResidualBlock을 시퀀스로 쌓은 모델
    """
    def __init__(self, channels=384, num_blocks=2):
        super(Residual_Blocks, self).__init__()
        
        # 원하는 개수만큼 ResidualBlock 쌓기
        blocks = []
        for _ in range(num_blocks):
            blocks.append(ResidualBlock(channels))
        
        # nn.Sequential로 여러 ResidualBlock을 묶어줌
        self.residual_layers = nn.Sequential(*blocks)

    def forward(self, x):
        # 입력 x를 순차적으로 ResidualBlock에 통과
        x = self.residual_layers(x)
        return x

class Residual_Adapter(nn.Module):
    def __init__(self):
        super(Residual_Adapter, self).__init__()
        
        self.dwconv = Double_DepthWise_Conv(384, 384)
        self.residual_blocks1 = Residual_Blocks(channels=384, num_blocks=1)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.residual_blocks2 = Residual_Blocks(channels=384, num_blocks=2)

    def forward(self, x, h, w, patch_h, patch_w):
        x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
        x = F.interpolate(x, size=(h//4, w//4), mode='bilinear', align_corners=False)
        x = self.dwconv(x)
        x = self.residual_blocks1(x)
        x = self.maxpool(x)
        x = self.residual_blocks2(x)
        x = F.interpolate(x, size=(patch_h, patch_w), mode='bilinear', align_corners=False)
        x = x.reshape((x.shape[0], x.shape[1], patch_h*patch_w)).permute(0, 2, 1)
        return x

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':
    model = Residual_Adapter()
    print(count_parameters(model))
    x = torch.randn((1, 3, 518, 518))
    out = model(x, 518//14, 518//14)
    print(out.size())


# class CNN_network(nn.Module):
#     def __init__(self):
#         super(CNN_network, self).__init__()
#         pretrained =  True
#         model = timm.create_model('mobilenetv2_100', pretrained=pretrained, features_only=True)
#         layers = [1,2,3,5,6]
#         chans = [16, 24, 32, 96, 160]
#         self.conv_stem = model.conv_stem
#         self.bn1 = model.bn1
#         self.act1 = nn.ReLU()

#         self.block0 = torch.nn.Sequential(*model.blocks[0:layers[0]])
#         self.block1 = torch.nn.Sequential(*model.blocks[layers[0]:layers[1]])
#         self.block2 = torch.nn.Sequential(*model.blocks[layers[1]:layers[2]])
#         self.block3 = torch.nn.Sequential(*model.blocks[layers[2]:layers[3]])
#         self.block4 = torch.nn.Sequential(*model.blocks[layers[3]:layers[4]])

#         self.deconv32_16 = Conv2x_IN(chans[4], chans[3], deconv=True, concat=True)
#         self.deconv16_8 = Conv2x_IN(chans[3]*2, chans[2], deconv=True, concat=True)
#         self.deconv8_4 = Conv2x_IN(chans[2]*2, chans[1], deconv=True, concat=True)
#         self.conv4 = BasicConv_IN(chans[1]*2, chans[1]*2, kernel_size=3, stride=1, padding=1)

#     def forward(self, x):
#         x = self.act1(self.bn1(self.conv_stem(x)))
#         x2 = self.block0(x)
#         x4 = self.block1(x2)
#         x8 = self.block2(x4)
#         x16 = self.block3(x8)
#         x32 = self.block4(x16)

#         x16 = self.deconv32_16(x32, x16)
#         x8 = self.deconv16_8(x16, x8)
#         x4 = self.deconv8_4(x8, x4)
#         x4 = self.conv4(x4)
#         return x4
    

# class BasicConv_IN(nn.Module):

#     def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, IN=True, relu=True, **kwargs):
#         super(BasicConv_IN, self).__init__()

#         self.relu = relu
#         self.use_in = IN
#         if is_3d:
#             if deconv:
#                 self.conv = nn.ConvTranspose3d(in_channels, out_channels, bias=False, **kwargs)
#             else:
#                 self.conv = nn.Conv3d(in_channels, out_channels, bias=False, **kwargs)
#             self.IN = nn.InstanceNorm3d(out_channels)
#         else:
#             if deconv:
#                 self.conv = nn.ConvTranspose2d(in_channels, out_channels, bias=False, **kwargs)
#             else:
#                 self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
#             self.IN = nn.InstanceNorm2d(out_channels)

#     def forward(self, x):
#         x = self.conv(x)
#         if self.use_in:
#             x = self.IN(x)
#         if self.relu:
#             x = nn.LeakyReLU()(x)#, inplace=True)
#         return x


# class Conv2x_IN(nn.Module):

#     def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, concat=True, keep_concat=True, IN=True, relu=True, keep_dispc=False):
#         super(Conv2x_IN, self).__init__()
#         self.concat = concat
#         self.is_3d = is_3d 
#         if deconv and is_3d: 
#             kernel = (4, 4, 4)
#         elif deconv:
#             kernel = 4
#         else:
#             kernel = 3

#         if deconv and is_3d and keep_dispc:
#             kernel = (1, 4, 4)
#             stride = (1, 2, 2)
#             padding = (0, 1, 1)
#             self.conv1 = BasicConv_IN(in_channels, out_channels, deconv, is_3d, IN=True, relu=True, kernel_size=kernel, stride=stride, padding=padding)
#         else:
#             self.conv1 = BasicConv_IN(in_channels, out_channels, deconv, is_3d, IN=True, relu=True, kernel_size=kernel, stride=2, padding=1)

#         if self.concat: 
#             mul = 2 if keep_concat else 1
#             self.conv2 = BasicConv_IN(out_channels*2, out_channels*mul, False, is_3d, IN, relu, kernel_size=3, stride=1, padding=1)
#         else:
#             self.conv2 = BasicConv_IN(out_channels, out_channels, False, is_3d, IN, relu, kernel_size=3, stride=1, padding=1)

#     def forward(self, x, rem):
#         x = self.conv1(x)
#         if x.shape != rem.shape:
#             x = F.interpolate(
#                 x,
#                 size=(rem.shape[-2], rem.shape[-1]),
#                 mode='nearest')
#         if self.concat:
#             x = torch.cat((x, rem), 1)
#         else: 
#             x = x + rem
#         x = self.conv2(x)
#         return x