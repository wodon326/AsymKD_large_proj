import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from huggingface_hub import PyTorchModelHubMixin, hf_hub_download

from depth_anything.blocks import FeatureFusionBlock, _make_scratch

from torchhub.facebookresearch_dinov2_main.dinov2.layers.block import Channel_Based_CrossAttentionBlock,CrossAttentionBlock,Block
from torchhub.facebookresearch_dinov2_main.dinov2.layers.mlp import Mlp

from AsymKD.cnn_module import CNN_network, SpatialAttentionExtractor, ChannelAttentionEnhancement

def _make_fusion_block(features, use_bn, size = None):
    return FeatureFusionBlock(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size,
    )


class DPTHead(nn.Module):
    def __init__(self, nclass, in_channels, features= 64, use_bn=False, out_channels= [48, 96, 192, 384], use_clstoken=False):
        super(DPTHead, self).__init__()
        
        self.nclass = nclass
        self.use_clstoken = use_clstoken
        


        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for out_channel in out_channels
        ])
        


        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(
                in_channels=out_channels[0],
                out_channels=out_channels[0],
                kernel_size=4,
                stride=4,
                padding=0),
            nn.ConvTranspose2d(
                in_channels=out_channels[1],
                out_channels=out_channels[1],
                kernel_size=2,
                stride=2,
                padding=0),
            nn.Identity(),
            nn.Conv2d(
                in_channels=out_channels[3],
                out_channels=out_channels[3],
                kernel_size=3,
                stride=2,
                padding=1)
        ])
        
        if use_clstoken:
            self.readout_projects = nn.ModuleList()
            for _ in range(len(self.projects)):
                self.readout_projects.append(
                    nn.Sequential(
                        nn.Linear(2 * in_channels, in_channels),
                        nn.GELU()))
        
        self.scratch = _make_scratch(
            out_channels,
            features,
            groups=1,
            expand=False,
        )

        self.scratch.stem_transpose = None
        
        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        head_features_1 = features
        head_features_2 = 32
        
        if nclass > 1:
            self.scratch.output_conv = nn.Sequential(
                nn.Conv2d(head_features_1, head_features_1, kernel_size=3, stride=1, padding=1),
                nn.ReLU(True),
                nn.Conv2d(head_features_1, nclass, kernel_size=1, stride=1, padding=0),
            )
        else:
            self.scratch.output_conv1 = nn.Conv2d(head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1)
            
            self.scratch.output_conv2 = nn.Sequential(
                nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
                nn.ReLU(True),
                nn.Conv2d(head_features_2, 1, kernel_size=1, stride=1, padding=0),
                nn.ReLU(True),
                nn.Identity(),
            )
            
    def forward(self, out_features, patch_h, patch_w):
        out = []
        for i, x in enumerate(out_features):
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
            
            # torch.save(x, f'layer{i}_Compress_feature.pt')

            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            out.append(x)


        layer_1, layer_2, layer_3, layer_4 = out
        

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)
        
        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
        
        out = self.scratch.output_conv1(path_1)
        out = F.interpolate(out, (int(patch_h * 14), int(patch_w * 14)), mode="bilinear", align_corners=True)
        out = self.scratch.output_conv2(out)
        
        return out
        
        
class AsymKD_kd_naive_dpt_latent1_cnn_hybrid(nn.Module):
    def __init__(self, encoder='vits', features= 64, out_channels= [48, 96, 192, 384], use_bn=False, use_clstoken=False, localhub=True):
        super(AsymKD_kd_naive_dpt_latent1_cnn_hybrid, self).__init__()
        
        assert encoder in ['vits', 'vitb', 'vitl']
        print('AsymKD_kd_naive_dpt_latent1_cnn_hybrid')

        # in case the Internet connection is not stable, please load the DINOv2 locally
        if localhub:
            self.pretrained = torch.hub.load('torchhub/facebookresearch_dinov2_main', 'dinov2_{:}14'.format(encoder), source='local', pretrained=False)
        else:
            self.pretrained = torch.hub.load('facebookresearch/dinov2', 'dinov2_{:}14'.format(encoder))

        dim = self.pretrained.blocks[0].attn.qkv.in_features

        self.depth_head = DPTHead(1, dim, features, use_bn, out_channels=out_channels, use_clstoken=use_clstoken)


        cnn_feat_channel = 48
        self.sam = SpatialAttentionExtractor()
        self.cam = ChannelAttentionEnhancement(cnn_feat_channel)

        self.cnn_module = CNN_network()

        self.mlp_layer = Mlp(
                in_features=cnn_feat_channel*3+dim,
                out_features= dim,
            )

        self.nomalize = NormalizeLayer()

    def forward(self, x):
        h, w = x.shape[-2:]
        
        intermediate_feature = self.pretrained.get_first_intermediate_layers(x, 4)
        patch_h, patch_w = h // 14, w // 14

        #CNN 모듈 및 CBAM 모듈을 통한 feature extract
        cnn_feature = self.cnn_module(x)
        cam_feat = self.cam(cnn_feature) * cnn_feature
        attn = self.sam(cam_feat)
        high_freq_feat = cnn_feature * attn
        low_freq_feat = cnn_feature * (1 - attn)
        cnn_concat_features = torch.cat((high_freq_feat, cnn_feature, low_freq_feat), dim=1)
        cnn_concat_features = F.interpolate(cnn_concat_features, size=(patch_h*2, patch_w*2), mode='bilinear', align_corners=False)
        cnn_concat_features = F.max_pool2d(cnn_concat_features, kernel_size=2)
        cnn_concat_features = cnn_concat_features.reshape(cnn_concat_features.shape[0], cnn_concat_features.shape[1], patch_h * patch_w).permute(0, 2, 1)


        #CNN feature와 small 모델 feature fusion
        fusion_feat = self.mlp_layer(torch.cat((cnn_concat_features, intermediate_feature), dim=2))


        features = self.pretrained.get_intermediate_layers_start_intermediate(fusion_feat, 3, return_class_token=False)

        depth = self.depth_head(features, patch_h, patch_w)
        depth = F.interpolate(depth, size=(h, w), mode="bilinear", align_corners=True)
        depth = F.relu(depth)
        depth = self.nomalize(depth) if self.training else depth

        if self.training:
            return depth, fusion_feat

        return depth
    
    def forward_val(self, x):
        h, w = x.shape[-2:]
        
        intermediate_feature = self.pretrained.get_first_intermediate_layers(x, 4)
        patch_h, patch_w = h // 14, w // 14

        #CNN 모듈 및 CBAM 모듈을 통한 feature extract
        cnn_feature = self.cnn_module(x)
        cam_feat = self.cam(cnn_feature) * cnn_feature
        attn = self.sam(cam_feat)
        high_freq_feat = cnn_feature * attn
        low_freq_feat = cnn_feature * (1 - attn)
        cnn_concat_features = torch.cat((high_freq_feat, cnn_feature, low_freq_feat), dim=1)
        cnn_concat_features = F.interpolate(cnn_concat_features, size=(patch_h*2, patch_w*2), mode='bilinear', align_corners=False)
        cnn_concat_features = F.max_pool2d(cnn_concat_features, kernel_size=2)
        cnn_concat_features = cnn_concat_features.reshape(cnn_concat_features.shape[0], cnn_concat_features.shape[1], patch_h * patch_w).permute(0, 2, 1)


        #CNN feature와 small 모델 feature fusion
        fusion_feat = self.mlp_layer(torch.cat((cnn_concat_features, intermediate_feature), dim=2))


        features = self.pretrained.get_intermediate_layers_start_intermediate(fusion_feat, 3, return_class_token=False)
        depth = self.depth_head(features, patch_h, patch_w)
        depth = F.interpolate(depth, size=(h, w), mode="bilinear", align_corners=True)
        depth = F.relu(depth)
        depth = self.nomalize(depth) if self.training else depth


        small_features = self.pretrained.get_intermediate_layers_start_intermediate(intermediate_feature, 3, return_class_token=False)
        small_depth = self.depth_head(small_features, patch_h, patch_w)
        small_depth = F.interpolate(small_depth, size=(h, w), mode="bilinear", align_corners=True)
        small_depth = F.relu(small_depth)
        small_depth = self.nomalize(small_depth) if self.training else small_depth

        return [depth, small_depth]
    
    def freeze_kd_naive_dpt_latent1_cnn_hybrid_style(self):
        
        # for i, (name, param) in enumerate(self.pretrained.named_parameters()):
        #     param.requires_grad = False

        for i, (name, param) in enumerate(self.depth_head.named_parameters()):
            param.requires_grad = False

        self.pretrained.freeze_last_n_blocks(n = 3)
        
    
    def load_backbone_from_ckpt(
        self,
        student_ckpt: str,
        device: torch.device,
    ):
        assert student_ckpt.endswith('.pth'), 'Please provide the path to the checkpoint file.'
        
        ckpt = torch.load(student_ckpt, map_location=device)
        model_state_dict = self.state_dict()
        new_state_dict = {
            k: v for k, v in ckpt.items() if k in model_state_dict
        }
        model_state_dict.update(new_state_dict)
        self.load_state_dict(model_state_dict)

    
        return None

    
    def load_ckpt(
        self,
        ckpt: str,
        device: torch.device
    ):
        assert ckpt.endswith('.pth'), 'Please provide the path to the checkpoint file.'
        
        ckpt = torch.load(ckpt, map_location=device)
        ckpt = ckpt['model_state_dict']
        model_state_dict = self.state_dict()
        new_state_dict = {}
        for k, v in ckpt.items():
            # 키 매핑 규칙을 정의
            new_key = k.replace('module.', '')  # 'module.'를 제거
            if new_key in model_state_dict:
                new_state_dict[new_key] = v

        model_state_dict.update(new_state_dict)
        self.load_state_dict(model_state_dict)
    
        return new_state_dict
    
class NormalizeLayer(nn.Module):
    def __init__(self):
        super(NormalizeLayer, self).__init__()
    
    def forward(self, x):
        min_val = x.amin(dim=(1, 2, 3), keepdim=True)  # 각 배치별 최소값
        max_val = x.amax(dim=(1, 2, 3), keepdim=True)  # 각 배치별 최대값
        x = (x - min_val) / (max_val - min_val + 1e-6)
        return x
