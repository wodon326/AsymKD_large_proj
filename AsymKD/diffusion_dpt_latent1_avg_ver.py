import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from functools import partial
from huggingface_hub import PyTorchModelHubMixin, hf_hub_download

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
        
        
class diffusion_dpt_latent1_avg_ver(nn.Module):
    def __init__(self, feature_generate_diffusion ,encoder='vits', features= 64, out_channels= [48, 96, 192, 384], use_bn=False, use_clstoken=False, localhub=True):
        super(diffusion_dpt_latent1_avg_ver, self).__init__()
        
        assert encoder in ['vits', 'vitb', 'vitl']
        print('diffusion_dpt_latent1_avg_ver')
        self.feature_generate_diffusion = feature_generate_diffusion


        # in case the Internet connection is not stable, please load the DINOv2 locally
        if localhub:
            self.pretrained = torch.hub.load('torchhub/facebookresearch_dinov2_main', 'dinov2_{:}14'.format(encoder), source='local', pretrained=False)
        else:
            self.pretrained = torch.hub.load('facebookresearch/dinov2', 'dinov2_{:}14'.format(encoder))

        
        for i, (name, param) in enumerate(self.pretrained.named_parameters()):
            param.requires_grad = False

        dim = self.pretrained.blocks[0].attn.qkv.in_features

        self.depth_head = DPTHead(1, dim, features, use_bn, out_channels=out_channels, use_clstoken=use_clstoken)
        for i, (name, param) in enumerate(self.depth_head.named_parameters()):
            param.requires_grad = False
        

        self.nomalize = NormalizeLayer()

    def forward(self, x):
        h, w = x.shape[-2:]
        
        student_intermediate_feature = self.pretrained.get_first_intermediate_layers(x, 4)
        patch_h, patch_w = h // 14, w // 14

        ######## To-Do : student_intermediate_feature를 input으로 diffusion으로 compress_feature 생성 #########
        
        batch_size = student_intermediate_feature.shape[0]
        cond_feature = rearrange(student_intermediate_feature, 'b n c -> (b n) c')
        compress_feature = self.feature_generate_diffusion.sample(cond_feature, batch_size=batch_size)
        compress_feature = rearrange(compress_feature, '(b n) c -> b n c', b=batch_size)

        #################################################################################

        features = self.pretrained.get_intermediate_layers_start_intermediate(compress_feature, 3, return_class_token=False)

        depth = self.depth_head(features, patch_h, patch_w)
        depth = F.interpolate(depth, size=(h, w), mode="bilinear", align_corners=True)
        depth = F.relu(depth)
        depth = self.nomalize(depth) if self.training else depth

        return depth
    
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


        for i, (name, param) in enumerate(self.pretrained.named_parameters()):
            param.requires_grad = False
        
        return None

class NormalizeLayer(nn.Module):
    def __init__(self):
        super(NormalizeLayer, self).__init__()
    
    def forward(self, x):
        min_val = x.min()
        max_val = x.max()
        x = (x - min_val) / (max_val - min_val + 1e-6)  # 작은 값을 더하여 0으로 나누는 것을 방지합니다.
        return x

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--encoder",
        default="vits",
        type=str,
        choices=["vits", "vitb", "vitl"],
    )
    args = parser.parse_args()
    
    model = DepthAnything.from_pretrained("LiheYoung/depth_anything_{:}14".format(args.encoder))
    
    print(model)
    