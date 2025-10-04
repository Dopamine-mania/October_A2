# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# --- 1. 基础模块  ---
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(1, out_ch),
            nn.GELU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(1, out_ch),
            nn.GELU()
        )
    def forward(self, x, t):
        h = self.conv1(x)
        time_emb = self.time_mlp(t).unsqueeze(-1).unsqueeze(-1)
        h = h + time_emb
        h = self.conv2(h)
        return h

# --- 2. 注意力模块 ---
# ... (SelfAttention and CrossAttention classes remain the same as the previous correct version) ...
class SelfAttention(nn.Module):
    def __init__(self, in_channels, n_head=4):
        super().__init__()
        self.n_head = n_head
        self.norm = nn.GroupNorm(1, in_channels)
        self.qkv = nn.Linear(in_channels, in_channels * 3)
        self.proj_out = nn.Linear(in_channels, in_channels)
    def forward(self, x):
        B, C, H, W = x.shape
        x_in = x
        x = self.norm(x)
        x = x.view(B, C, H * W).permute(0, 2, 1)
        qkv = self.qkv(x).view(B, -1, self.n_head, (C * 3) // self.n_head)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.permute(0, 2, 1, 3).reshape(B * self.n_head, -1, C // self.n_head)
        k = k.permute(0, 2, 1, 3).reshape(B * self.n_head, -1, C // self.n_head)
        v = v.permute(0, 2, 1, 3).reshape(B * self.n_head, -1, C // self.n_head)
        attn = torch.bmm(q, k.transpose(-1, -2)) * ((C // self.n_head) ** -0.5)
        attn = F.softmax(attn, dim=-1)
        out = torch.bmm(attn, v)
        out = out.reshape(B, self.n_head, -1, C // self.n_head).permute(0, 2, 1, 3).reshape(B, -1, C)
        out = self.proj_out(out).permute(0, 2, 1).view(B, C, H, W)
        return x_in + out

class CrossAttention(nn.Module):
    def __init__(self, in_channels, context_dim, n_head=4):
        super().__init__()
        self.n_head = n_head
        self.norm_x = nn.GroupNorm(1, in_channels)
        self.norm_context = nn.LayerNorm(context_dim)
        self.query = nn.Linear(in_channels, in_channels)
        self.key = nn.Linear(context_dim, in_channels)
        self.value = nn.Linear(context_dim, in_channels)
        self.proj_out = nn.Linear(in_channels, in_channels)
    def forward(self, x, context):
        x_in = x
        B, C, H, W = x.shape
        x = self.norm_x(x)
        x = x.view(B, C, H * W).permute(0, 2, 1)
        context = self.norm_context(context).unsqueeze(1)
        q = self.query(x)
        k = self.key(context)
        v = self.value(context)
        q = q.view(B, -1, self.n_head, C // self.n_head).permute(0, 2, 1, 3).reshape(B * self.n_head, -1, C // self.n_head)
        k = k.view(B, -1, self.n_head, C // self.n_head).permute(0, 2, 1, 3).reshape(B * self.n_head, -1, C // self.n_head)
        v = v.view(B, -1, self.n_head, C // self.n_head).permute(0, 2, 1, 3).reshape(B * self.n_head, -1, C // self.n_head)
        attn = torch.bmm(q, k.transpose(-1, -2)) * ((C // self.n_head) ** -0.5)
        attn = F.softmax(attn, dim=-1)
        out = torch.bmm(attn, v)
        out = out.reshape(B, self.n_head, -1, C // self.n_head).permute(0, 2, 1, 3).reshape(B, -1, C)
        out = self.proj_out(out).permute(0, 2, 1).view(B, C, H, W)
        return x_in + out

# --- 3. U-Net核心架构  ---
class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, time_dim=256, context_dim=256, enable_self_attention=True, enable_cross_attention=True):
        super().__init__()
        self.enable_self_attention = enable_self_attention
        self.enable_cross_attention = enable_cross_attention
        
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim), nn.Linear(time_dim, time_dim), nn.GELU()
        )
        
        # Encoder
        self.inc = Block(in_channels, 64, time_dim)
        self.down1_pool = nn.MaxPool2d(2)
        self.down1_conv = Block(64, 128, time_dim)
        self.down2_pool = nn.MaxPool2d(2)
        self.down2_conv = Block(128, 256, time_dim)
        
        # Bottleneck
        self.bot1 = Block(256, 512, time_dim)
        if self.enable_self_attention: self.sa = SelfAttention(512)
        if self.enable_cross_attention: self.ca_bot = CrossAttention(512, context_dim)
        self.bot2 = Block(512, 512, time_dim)

        # Decoder
        self.up1_trans = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up1_conv = Block(384, 256, time_dim) # Concat: 256 (from upsample) + 256 (from skip)

        self.up2_trans = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up2_conv = Block(192, 128, time_dim) # Concat: 128 (from upsample) + 128 (from skip)
        
        self.outc = nn.Conv2d(128, out_channels, kernel_size=1)
        
        if self.enable_cross_attention:
            self.ca_up1 = CrossAttention(256, context_dim)
            self.ca_up2 = CrossAttention(128, context_dim)

    def forward(self, x, t, context):
        
        
        # Down path (Encoder)
        x1 = self.inc(x, t)
        x2 = self.down1_pool(x1)
        x2 = self.down1_conv(x2, t)
        x3 = self.down2_pool(x2)
        x3 = self.down2_conv(x3, t)

        # Bottleneck
        x_bottle = self.bot1(x3, t)
        if self.enable_self_attention: x_bottle = self.sa(x_bottle)
        if self.enable_cross_attention: x_bottle = self.ca_bot(x_bottle, context)
        x_bottle = self.bot2(x_bottle, t)

        # Up path (Decoder)
        x = self.up1_trans(x_bottle)
        x = torch.cat([x, x2], dim=1)
        x = self.up1_conv(x, t)
        if self.enable_cross_attention: x = self.ca_up1(x, context)

        x = self.up2_trans(x)
        x = torch.cat([x, x1], dim=1)
        x = self.up2_conv(x, t)
        if self.enable_cross_attention: x = self.ca_up2(x, context)
        
        # Final output
        x = F.interpolate(x, size=x1.shape[-2:], mode='bilinear', align_corners=False)
        return self.outc(x)


# --- 4. 模型外壳  ---
class DiffusionModel(nn.Module):
    def __init__(self, enable_self_attention=True, enable_cross_attention=True):
        super().__init__()
        self.cond_encoder = nn.Sequential(
            nn.Linear(1, 128),
            nn.GELU(),
            nn.Linear(128, 256)
        )
        self.image_unet = UNet(
             in_channels=1, out_channels=1, time_dim=256, context_dim=256,
             enable_self_attention=enable_self_attention,
             enable_cross_attention=enable_cross_attention
        ) 
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(256),
            nn.Linear(256, 256),
            nn.GELU()
        )
    def forward(self, y_img, t, cond):
        time_emb = self.time_mlp(t)
        cond_emb = self.cond_encoder(cond)
        eps_pred = self.image_unet(y_img, time_emb, cond_emb)
        return eps_pred