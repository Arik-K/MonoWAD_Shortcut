import torch.nn as nn
import torch
import os

from visualDet3D.networks.backbones.dla import dla102
from visualDet3D.networks.backbones.dlaup import DLAUp
from visualDet3D.networks.detectors.dfe import DepthAwareFE
from visualDet3D.networks.detectors.dpe import DepthAwarePosEnc
from visualDet3D.networks.detectors.dtr import DepthAwareTransformer
from visualDet3D.networks.detectors.denoising_diffusion_pytorch import Unet, ShortcutDiffusion  ##GaussianDiffusion
from visualDet3D.networks.detectors.wc import WeatherCodebook
from visualDet3D.networks.heads.losses import Loss
            

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
        #self.num_timesteps = 15
        self.codebook = WeatherCodebook(4096, self.output_channel_num, 256)
        # 2. Initialize the Shortcut Engine
        self.diffusion_init()

        #4. CRITICAL FIX: Initialize the Head Loss
        # This allows self.loss(feat, P2, annotations) to work in forward()
        if 'loss_cfg' in backbone_arguments:
            self.loss = Loss(backbone_arguments['loss_cfg'])
        else:
            print("Warning: 'loss_cfg' not found in arguments. Training will crash if head loss is needed.")

        # 3. Load & Freeze Logic (For 4x A40 Training)
        # Assuming the weight file is in your workdirs folder 
        pretrained_path = 'workdirs/MonoWAD/checkpoint/MonoWAD_3D.pth' # change based on
        if os.path.exists(pretrained_path):
            self.load_and_freeze_teacher(pretrained_path)

    def diffusion_init(self):
        self.unet = Unet(
            dim=64,
            dim_mults=(1, 2, 4),
            full_attn=(False, False, True),
            channels=256,
            flash_attn=True
        )

        self.diffusion = ShortcutDiffusion(
            self.unet,
            image_size=(36, 160),
            #timesteps=self.num_timesteps,    # number of steps
            max_discretization_steps=16 # Grid resolution for consistency
        )

    

    # def predict(self, x, t: int, codebook=None):
    #     b, *_, device = *x.shape, x.device
    #     batched_times = torch.full((b,), t, device=device, dtype=torch.long)
    #     model_mean, _, model_log_variance, x_start = self.diffusion.p_mean_variance(
    #         x=x, t=batched_times, codebook=codebook, clip_denoised=True)
    #     noise = x - x_start if t > 0 else 0
    #     pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
    #     return pred_img, x_start

    # def enhancing_feature_representation(self, fog_feat, codebook=None):
    #     diff_feat = [fog_feat]
    #     for t in reversed(range(self.num_timesteps)):
    #         fog_feat, x_start = self.predict(fog_feat, t, codebook)
    #         diff_feat.append(fog_feat)
        
    #     diffusion_feat = fog_feat
    #     return diffusion_feat

    # def weather_adaptive_diffusion(self, origin_feat, noised_feat=None, codebook=None):
    #     if noised_feat is not None:
    #         loss = self.diffusion(origin_feat, noised_feat, codebook)
    #         return loss, self.enhancing_feature_representation(origin_feat, codebook)
    #     else:
    #         return self.enhancing_feature_representation(origin_feat, codebook)

    # def forward(self, x):
    #     training = x["training"]
    #     origin_feat = self.backbone(x['image'])
    #     origin_feat = self.neck(origin_feat[self.first_level:])
        
    #     if training:
    #         foggy_feat = self.backbone(x['foggy'])
    #         foggy_feat = self.neck(foggy_feat[self.first_level:])
    #         weather_reference_feat, l_ckr = self.codebook(origin_feat, foggy_feat)
    #         l_wae, x = self.weather_adaptive_diffusion(origin_feat, foggy_feat, weather_reference_feat)
    #         l_proposed = l_wae + l_ckr
    #     else:
    #         weather_reference_feat = self.codebook(origin_feat)
    #         x = self.weather_adaptive_diffusion(origin_feat, codebook=weather_reference_feat)
    #     N, C, H, W = x.shape

    #     depth, depth_guide, depth_feat = self.dfe(x)
        
    #     depth_feat = depth_feat.permute(0, 2, 3, 1).view(N, H*W, C)
        
    #     depth_guide = depth_guide.argmax(1)
    #     depth_emb = self.depth_embed(depth_guide).view(N, H*W, C)
    #     depth_emb = self.dpe(depth_emb, (H,W))
        
    #     img_feat = x + self.img_conv(x)
    #     img_feat = img_feat.view(N, H*W, C)
    #     feat = self.dtr(depth_feat, img_feat, depth_emb)
    #     feat = feat.permute(0, 2, 1).view(N,C,H,W)
    #     if training:
    #         return feat, depth, l_proposed
    #     else:
    #         return feat, depth


    def load_and_freeze_teacher(self, path):
        """
        Loads pretrained weights for everything EXCEPT the diffusion engine,
        then freezes those layers to focus training on the Shortcut.
        """
        checkpoint = torch.load(path, map_location='cpu')
        
        # Determine weight key (handles different checkpoint formats)
        state_dict = checkpoint.get('model_state', checkpoint.get('state_dict', checkpoint))

        model_dict = self.state_dict()
        # Filter out diffusion/unet keys to avoid mismatch with the new Shortcut logic
        pretrained_dict = {k: v for k, v in state_dict.items() 
                          if k in model_dict and 'diffusion' not in k and 'unet' not in k}
        
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)
        
        # Freeze Teacher Components
        for name, param in self.named_parameters():
            if 'diffusion' in name or 'unet' in name:
                param.requires_grad = True  # Train the new Shortcut
            else:
                param.requires_grad = False # Freeze everything else
        
        print(f"Teacher weights loaded. Shortcut engine is trainable.")

    def forward(self, x):
    """
    Forward pass supporting both training and inference. 
    
    MonoWAD_3D calls this with:  dict(image=..., P2=..., foggy=..., training=True/False)
    """
    # Handle dict input (from MonoWAD_3D)
    if isinstance(x, dict):
        training = x. get("training", False)
        images = x['image']
        foggy_images = x.get('foggy', None)
    # Handle list input (direct call from trainer - legacy support)
    elif isinstance(x, list):
        images = x[0]
        annotations = x[1]
        P2 = x[2]
        depth_gt = x[3]
        foggy_images = x[4]
        training = True
    else: 
        raise ValueError(f"Unexpected input type: {type(x)}")

    origin_feat = self. backbone(images)
    origin_feat = self.neck(origin_feat[self.first_level: ])
    
    if training:
        foggy_feat = self.backbone(foggy_images)
        foggy_feat = self. neck(foggy_feat[self.first_level:])
        
        # Weather Codebook
        weather_reference_feat, l_ckr = self.codebook(origin_feat, foggy_feat)
        
        # Shortcut Diffusion Loss
        diffusion_losses = self.diffusion(origin_feat, foggy_feat, weather_reference_feat)
        l_wae = diffusion_losses['loss_total']
        l_proposed = l_wae + l_ckr
        
        # Enhanced features for detection
        with torch.no_grad():
            x_enhanced = self.diffusion.sample(origin_feat, weather_reference_feat)
    else:
        weather_reference_feat = self.codebook(origin_feat)
        x_enhanced = self.diffusion.sample(origin_feat, weather_reference_feat)

    # Depth-aware processing
    N, C, H, W = x_enhanced.shape
    depth, depth_guide, depth_feat = self. dfe(x_enhanced)
    
    depth_feat = depth_feat.permute(0, 2, 3, 1).view(N, H*W, C)
    depth_guide = depth_guide.argmax(1)
    depth_emb = self.depth_embed(depth_guide).view(N, H*W, C)
    depth_emb = self. dpe(depth_emb, (H, W))
    
    img_feat = x_enhanced + self.img_conv(x_enhanced)
    img_feat = img_feat.view(N, H*W, C)
    feat = self. dtr(depth_feat, img_feat, depth_emb)
    feat = feat.permute(0, 2, 1).view(N, C, H, W)
    
    if training:
        # Return format expected by MonoWAD_3D. train_forward()
        # MonoWAD_3D expects: outputs[0]=feat, outputs[1]=depth, outputs[2]=l_proposed, outputs[3]=extra_logs
        return feat, depth, l_proposed, diffusion_losses
    else:
        return feat, depth
        
        #     return feat, depth, l_proposed
        # else:
        #     return feat, depth
