

import torch
import torch.nn as nn
import timm
import torch.nn.functional as F


class ChannelAttentionEnhancement(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttentionEnhancement, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttentionExtractor(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttentionExtractor, self).__init__()

        self.samconv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.samconv(x)
        return self.sigmoid(x)
    

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
    
class triple_DepthWise_Conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 1),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, groups=in_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, groups=out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, groups=out_channels)
        )
    
    def forward(self, x):
        return self.conv(x)
    
class Double_DepthWise_Conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 1),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, groups=in_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, groups=out_channels)
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
    def __init__(self, channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, stride=stride)
        # self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        # self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()
        self.stride = stride
        if stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=1, stride=stride),
                # nn.BatchNorm2d(channels)
            )
            

    def forward(self, x):
        # skip connection
        identity = x.clone()
        # out = self.relu(self.bn1(self.conv1(x)))
        # out = self.bn2(self.conv2(out))
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        if self.stride != 1:
            identity = self.downsample(identity)
        out += identity  # Residual 연결
        out = self.relu(out)
        return out

class Residual_Blocks(nn.Module):
    """
    channels=384인 입력에 대해 여러 개의 ResidualBlock을 시퀀스로 쌓은 모델
    """
    def __init__(self, channels=384, num_blocks=2, stride=1):
        super(Residual_Blocks, self).__init__()
        
        # 원하는 개수만큼 ResidualBlock 쌓기
        blocks = []
        for _ in range(num_blocks):
            blocks.append(ResidualBlock(channels,stride))
            stride = 1
        
        # nn.Sequential로 여러 ResidualBlock을 묶어줌
        self.residual_layers = nn.Sequential(*blocks)

    def forward(self, x):
        # 입력 x를 순차적으로 ResidualBlock에 통과
        x = self.residual_layers(x)
        return x

class UNet_CBAM_adapter(nn.Module):
    def __init__(self,  output_channels=384):
        super().__init__()
        base_channels = output_channels//16
        self.inc = DoubleConv(3, base_channels)
        self.pre_down1 = Down(base_channels, base_channels*2)
        self.pre_down2 = Down(base_channels*2, base_channels*2)

        base_channels = output_channels//8
        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8)
        
        self.up2 = Up(base_channels * 8, base_channels * 4)
        self.up3 = Up(base_channels * 4, base_channels * 2)
        self.up4 = Up(base_channels * 2, base_channels*2)
        
        self.unet_final_conv = nn.Conv2d(base_channels*2, base_channels*2, kernel_size=3, padding=1)
        base_channels = output_channels//4

        # self.cbam = nn.Conv2d(output_channels, output_channels*3, kernel_size=3, padding=1)

        self.cam1 = ChannelAttentionEnhancement(base_channels)
        self.sam1 = SpatialAttentionExtractor()

        self.down_conv1 = nn.Sequential(
            nn.Conv2d(base_channels*3, base_channels*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels*2),
            nn.ReLU()
        )
        self.conv1 = Residual_Blocks(channels=base_channels*2, num_blocks=2, stride=1)

        self.refine_conv2 = nn.Conv2d(base_channels*2, base_channels*2, kernel_size=3, padding=1)
        
        
        self.cam2 = ChannelAttentionEnhancement(base_channels*2)
        self.sam2 = SpatialAttentionExtractor()
        self.down_conv2 = nn.Sequential(
            nn.Conv2d(base_channels*6, base_channels*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels*4),
            nn.ReLU()
        )
        self.conv2 = Residual_Blocks(channels=base_channels*4, num_blocks=2, stride=1)


        self.conv3 = Residual_Blocks(channels=base_channels*4, num_blocks=2, stride=2)
        self.final_conv = nn.Conv2d(base_channels*4, base_channels*4, kernel_size=3, padding=1)


        
    def forward(self, x, patch_h, patch_w):
        # U-Net
        x1 = self.inc(x)
        x1 = self.pre_down1(x1)
        x1 = self.pre_down2(x1)

        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        x = self.up2(x4, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        x = self.unet_final_conv(x)
        
        # cbam_features = self.cbam(x)
        # CBAM
        cam_feat = self.cam1(x) * x
        attn = self.sam1(cam_feat)
        high_freq_feat = x * attn
        low_freq_feat = x * (1 - attn)
        cbam_features = torch.cat((high_freq_feat, x, low_freq_feat), dim=1)
        x = self.down_conv1(cbam_features)

        # Refinement
        x = self.conv1(x)

        # Downsample
        x = F.interpolate(x, size=(patch_h*2, patch_w*2), mode='bilinear', align_corners=True)
        x = self.refine_conv2(x)
        cam_feat = self.cam2(x) * x
        attn = self.sam2(cam_feat)
        high_freq_feat = x * attn
        low_freq_feat = x * (1 - attn)
        cbam_features = torch.cat((high_freq_feat, x, low_freq_feat), dim=1)
        x = self.down_conv2(cbam_features)
        x = self.conv2(x)

        x = self.conv3(x)
        x = self.final_conv(x)

        x = x.reshape((x.shape[0], x.shape[1], patch_h*patch_w)).permute(0, 2, 1)

        return x
    
    
class UNet_CBAM_adapter_for_trans(nn.Module):
    def __init__(self,  output_channels=384):
        super().__init__()
        base_channels = output_channels//16
        self.inc = DoubleConv(3, base_channels)
        self.pre_down1 = Down(base_channels, base_channels*2)
        self.pre_down2 = Down(base_channels*2, base_channels*2)

        base_channels = output_channels//8
        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8)
        
        self.up2 = Up(base_channels * 8, base_channels * 4)
        self.up3 = Up(base_channels * 4, base_channels * 2)
        self.up4 = Up(base_channels * 2, base_channels*2)
        
        self.unet_final_conv = nn.Conv2d(base_channels*2, base_channels*2, kernel_size=3, padding=1)
        base_channels = output_channels//4

        # self.cbam = nn.Conv2d(output_channels, output_channels*3, kernel_size=3, padding=1)

        self.cam = ChannelAttentionEnhancement(base_channels)
        self.sam = SpatialAttentionExtractor()

        self.conv1 = Residual_Blocks(channels=base_channels*3, num_blocks=2, stride=1)
        
        self.up_conv = nn.Sequential(
            nn.Conv2d(base_channels*3, base_channels*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels*4),
            nn.ReLU()
        )
        self.conv2 = Residual_Blocks(channels=base_channels*4, num_blocks=2, stride=1)

        self.final_conv = nn.Sequential(
            nn.Conv2d(base_channels*4, base_channels*4, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(base_channels*4),
            nn.ReLU(),
            nn.Conv2d(base_channels*4, base_channels*4, kernel_size=3, padding=1)
        )
        
        


        
    def forward(self, x, patch_h, patch_w):
        # U-Net
        x1 = self.inc(x)
        x1 = self.pre_down1(x1)
        x1 = self.pre_down2(x1)

        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        x = self.up2(x4, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        x = self.unet_final_conv(x)
        
        # cbam_features = self.cbam(x)
        # CBAM
        cam_feat = self.cam(x) * x
        attn = self.sam(cam_feat)
        high_freq_feat = x * attn
        low_freq_feat = x * (1 - attn)
        cbam_features = torch.cat((high_freq_feat, x, low_freq_feat), dim=1)

        # Refinement
        x = self.conv1(cbam_features)

        # Downsample
        x = self.up_conv(x)
        x = F.interpolate(x, size=(patch_h*2, patch_w*2), mode='bilinear', align_corners=True)
        x = self.conv2(x)

        x = self.final_conv(x)

        x = x.reshape((x.shape[0], x.shape[1], patch_h*patch_w)).permute(0, 2, 1)

        return x
    
    
class UNet_CBAM_adapter_for_cnn_trans_hybrid(nn.Module):
    def __init__(self, base_channels=64, output_channels=128):
        super().__init__()
        self.inc = DoubleConv(3, base_channels//2)
        self.pre_down1 = Down(base_channels//2, base_channels//2)
        self.pre_down2 = Down(base_channels//2, base_channels)

        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.up3 = Up(base_channels * 4, base_channels * 2)
        self.up4 = Up(base_channels * 2, output_channels)
        
        self.final_conv = nn.Conv2d(output_channels, output_channels, kernel_size=1)

        # self.cbam = nn.Conv2d(output_channels, output_channels*3, kernel_size=3, padding=1)

        self.cam = ChannelAttentionEnhancement(output_channels)
        self.sam = SpatialAttentionExtractor()

        output_channels = output_channels * 3
        self.pointwise = nn.Conv2d(output_channels, output_channels, kernel_size=1)
        self.dwconv = nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1, groups=output_channels)
        self.dwconv_refine = Double_DepthWise_Conv(output_channels, output_channels)

        self.maxpool = nn.MaxPool2d(2)
        self.dwconv_down1 = Double_DepthWise_Conv(output_channels, output_channels)
        self.dwconv_down2 = Double_DepthWise_Conv(output_channels, output_channels)



        
    def forward(self, x, patch_h, patch_w):
        # U-Net
        x = self.inc(x)
        x_2 = self.pre_down1(x)
        x_4 = self.pre_down2(x_2)

        x_8 = self.down1(x_4)
        x_16 = self.down2(x_8)
        x = self.up3(x_16, x_8)
        x = self.up4(x, x_4)
        
        x = self.final_conv(x)
        
        # cbam_features = self.cbam(x)
        # CBAM
        cam_feat = self.cam(x) * x
        attn = self.sam(cam_feat)
        high_freq_feat = x * attn
        low_freq_feat = x * (1 - attn)
        cbam_features = torch.cat((high_freq_feat, x, low_freq_feat), dim=1)

        # Refinement
        x = self.pointwise(cbam_features)
        x = self.dwconv(x)
        x = self.dwconv_refine(x)
        # Downsample
        x = self.maxpool(x)
        x = self.dwconv_down1(x)
        x = self.dwconv_down2(x)

        x = F.interpolate(x, size=(patch_h, patch_w), mode='bilinear', align_corners=True)
        x = x.reshape((x.shape[0], x.shape[1], patch_h*patch_w)).permute(0, 2, 1)

        return x





def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':
    model = UNet_CBAM_adapter(32, 128)
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