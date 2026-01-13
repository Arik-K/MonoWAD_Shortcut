import torch.nn as nn
import torch

from visualDet3D.networks.backbones.dla import dla102
from visualDet3D.networks.backbones.dlaup import DLAUp
from visualDet3D.networks.detectors.dfe import DepthAwareFE
from visualDet3D.networks.detectors.dpe import DepthAwarePosEnc
from visualDet3D.networks.detectors.dtr import DepthAwareTransformer
# Ensure this imports the NEW class you just pasted
from visualDet3D.networks.detectors.denoising_diffusion_pytorch import Unet, ShortcutDiffusion 
from visualDet3D.networks.detectors.wc import WeatherCodebook

class MonoWAD(nn.Module):
    def __init__(self, backbone_arguments=dict()):
        super(MonoWAD, self).__init__()
        self.backbone = dla102(pretrained=True, return_levels=True)
        channels = self.backbone.channels
        self.first_level = 3
        scales = [2**i for i in range(len(channels[self.first_level:]))]
        self.neck = DLAUp(channels[self.first_level:], scales_list=scales)

        self.output_channel_num = 256
        self.dpe = DepthAwarePosEnc(self.output_channel_num)
        self.depth_embed = nn.Embedding(100, self.output_channel_num)
        self.dtr = DepthAwareTransformer(self.output_channel_num)
        self.dfe = DepthAwareFE(self.output_channel_num)
        self.img_conv = nn.Conv2d(self.output_channel_num, self.output_channel_num, kernel_size=3, padding=1)
        
        # CHANGED: Replaced fixed timestep with codebook init only
        self.codebook = WeatherCodebook(4096, self.output_channel_num, 256)
        self.diffusion_init()

    def diffusion_init(self):
        # CHANGED: Unet initialized with updated args if needed (default is usually fine)
        self.unet = Unet(
            dim=64,
            dim_mults=(1, 2, 4),
            full_attn=(False, False, True),
            channels=256,
            flash_attn=True
        )

        # CHANGED: ShortcutDiffusion with M=128
        self.diffusion = ShortcutDiffusion(
            self.unet,
            image_size=(36, 160),
            max_discretization_steps=128
        )

    # CHANGED: Removed 'enhancing_feature_representation' loop (no longer needed)

    def forward(self, x):
        training = x["training"]
        origin_feat = self.backbone(x['image'])
        origin_feat = self.neck(origin_feat[self.first_level:])
        
        if training:
            foggy_feat = self.backbone(x['foggy'])
            foggy_feat = self.neck(foggy_feat[self.first_level:])
            
            # 1. Codebook Loss
            weather_reference_feat, l_ckr = self.codebook(origin_feat, foggy_feat)
            
            # 2. Shortcut Diffusion Loss (Joint Optimization)
            l_wae = self.diffusion(origin_feat, foggy_feat, weather_reference_feat)
            l_proposed = l_wae + l_ckr
            
            # 3. Prepare features for Head (1-step Clean)
            with torch.no_grad():
                enhanced_feat = self.diffusion.sample(foggy_feat, weather_reference_feat)
            x_det = enhanced_feat
            
        else:
            # Inference: 1-step Clean
            weather_reference_feat = self.codebook(origin_feat)
            x_det = self.diffusion.sample(origin_feat, weather_reference_feat)

        # Detection Head Logic (Unchanged)
        N, C, H, W = x_det.shape
        depth, depth_guide, depth_feat = self.dfe(x_det)
        
        depth_feat = depth_feat.permute(0, 2, 3, 1).view(N, H*W, C)
        depth_guide = depth_guide.argmax(1)
        depth_emb = self.depth_embed(depth_guide).view(N, H*W, C)
        depth_emb = self.dpe(depth_emb, (H,W))
        
        img_feat = x_det + self.img_conv(x_det)
        img_feat = img_feat.view(N, H*W, C)
        feat = self.dtr(depth_feat, img_feat, depth_emb)
        feat = feat.permute(0, 2, 1).view(N,C,H,W)
        
        if training:
            return feat, depth, l_proposed
        else:
            return feat, depth