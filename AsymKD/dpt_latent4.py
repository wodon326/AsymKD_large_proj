import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from depth_anything.blocks import FeatureFusionBlock, _make_scratch

from torchhub.facebookresearch_dinov2_main.dinov2.layers.block import Channel_Based_CrossAttentionBlock,CrossAttentionBlock,Block
from torchhub.facebookresearch_dinov2_main.dinov2.layers.mlp import Mlp

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

class NormalizeLayer(nn.Module):
    def __init__(self):
        super(NormalizeLayer, self).__init__()
    
    def forward(self, x):
        min_val = x.min()
        max_val = x.max()
        x = (x - min_val) / (max_val - min_val + 1e-6)  # 작은 값을 더하여 0으로 나누는 것을 방지합니다.
        return x


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
        
        
class AsymKD_compress(nn.Module):
    def __init__(
        self, 
        student_ckpt: str='depth_anything_vits14.pth',
        teacher_ckpt: str='depth_anything_vitl14.pth',
        features: int=64,
        out_channels: list[int]=[48, 96, 192, 384], 
        use_bn: bool=False, 
        use_clstoken: bool=False, 
        localhub: bool=True
    ):
        super(AsymKD_compress, self).__init__()
        
        student_encoder = 'vits'
        teacher_encoder = 'vitl'
        
        # in case the Internet connection is not stable, please load the DINOv2 locally
        if localhub:
            self.teacher_pretrained = torch.hub.load('torchhub/facebookresearch_dinov2_main', 'dinov2_{:}14'.format(teacher_encoder), source='local', pretrained=False).eval()
        else:
            self.teacher_pretrained = torch.hub.load('facebookresearch/dinov2', 'dinov2_{:}14'.format(teacher_encoder)).eval()

        # in case the Internet connection is not stable, please load the DINOv2 locally
        if localhub:
            self.pretrained = torch.hub.load('torchhub/facebookresearch_dinov2_main', 'dinov2_{:}14'.format(student_encoder), source='local', pretrained=False).eval()
        else:
            self.pretrained = torch.hub.load('facebookresearch/dinov2', 'dinov2_{:}14'.format(student_encoder)).eval()

        dim = self.pretrained.blocks[0].attn.qkv.in_features
        
        depth = 1
        drop_path_rate=0.0
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        in_channels_384 = 384
        in_channels_1024 = 1024

        self.Projects_layers_Channel_based_CrossAttn_Block = nn.ModuleList([
            Channel_Based_CrossAttentionBlock(
                dim_q=in_channels_384,
                dim_kv=in_channels_1024,
                num_heads=6,
                mlp_ratio=4,
                qkv_bias=True,
                proj_bias=True,
                ffn_bias=True,
                drop_path=dpr[0],
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                act_layer=nn.GELU,
                ffn_layer=Mlp,
                init_values=None,
            )
         for _ in range(4)])
        
        self.Projects_layers_Cross = nn.ModuleList([
            CrossAttentionBlock(
                dim=in_channels_384,
                num_heads=6,
                mlp_ratio=4,
                qkv_bias=True,
                proj_bias=True,
                ffn_bias=True,
                drop_path=dpr[0],
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                act_layer=nn.GELU,
                ffn_layer=Mlp,
                init_values=None,
            ) for _ in range(4)])

        self.Projects_layers_Self = nn.ModuleList([
            Block(
                dim=in_channels_384,
                num_heads=6,
                mlp_ratio=4,
                qkv_bias=True,
                proj_bias=True,
                ffn_bias=True,
                drop_path=dpr[0],
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                act_layer=nn.GELU,
                ffn_layer=Mlp,
                init_values=None,
            )
         for _ in range(4)])

        self.depth_head = DPTHead(1, dim, features, use_bn, out_channels=out_channels, use_clstoken=use_clstoken)
        for i, (name, param) in enumerate(self.depth_head.named_parameters()):
            param.requires_grad = False
        
        self.nomalize = NormalizeLayer()

    def forward(self, x):
        h, w = x.shape[-2:]
        self.teacher_pretrained.eval()
        self.pretrained.eval()

        teacher_features = self.teacher_pretrained.get_intermediate_layers(x, 4, return_class_token=False)
        student_features = self.pretrained.get_intermediate_layers(x, 4, return_class_token=False)
    
        patch_h, patch_w = h // 14, w // 14

        proj_feature = []
        for teacher_feat, student_feat, Channel_CrossAttn, CrossAtten, SelfAttn in zip(teacher_features,student_features,self.Projects_layers_Channel_based_CrossAttn_Block,self.Projects_layers_Cross, self.Projects_layers_Self):
            channel_proj_feat = Channel_CrossAttn(student_feat,teacher_feat)
            feat = CrossAtten(student_feat,channel_proj_feat)
            feat = SelfAttn(feat)
            proj_feature.append(feat)

        depth = self.depth_head(proj_feature, patch_h, patch_w)
        depth = F.interpolate(depth, size=(h, w), mode="bilinear", align_corners=True)
        depth = F.relu(depth)
        depth = self.nomalize(depth) if self.training else depth

        return depth
    
    def load_backbone_from_ckpt(
        self,
        student_ckpt: str,
        teacher_ckpt: str,
        device: torch.device,
    ):
        assert student_ckpt.endswith('.pth') and teacher_ckpt.endswith('.pth'), 'Please provide the path to the checkpoint file.'
        
        ckpt = torch.load(student_ckpt, map_location=device)
        model_state_dict = self.state_dict()
        new_state_dict = {
            k: v for k, v in ckpt.items() if k in model_state_dict
        }
        model_state_dict.update(new_state_dict)
        self.load_state_dict(model_state_dict)

        ckpt = torch.load(teacher_ckpt, map_location=device)
        new_state_dict = {}
        for k, v in ckpt.items():
            if('depth_head' in k):
                continue
            # 키 매핑 규칙을 정의
            new_key = k.replace('pretrained', 'teacher_pretrained')  # 'module.'를 제거
            if new_key in model_state_dict:
                new_state_dict[new_key] = v

        model_state_dict.update(new_state_dict)
        self.load_state_dict(model_state_dict)

        for i, (name, param) in enumerate(self.teacher_pretrained.named_parameters()):
            param.requires_grad = False

        for i, (name, param) in enumerate(self.pretrained.named_parameters()):
            param.requires_grad = False
        
        return None