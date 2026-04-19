
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from PIL import Image
from torchvision.utils import save_image
import torchvision.transforms as transforms
import cv2
import imageio.v3 as iio
import cam_param
import os
import numpy as np
import torch.nn.functional as F
import glob
import random

class VGGPerceptual(nn.Module):
    """

    conv3_3까지의 feature map을 사용한 content-loss
    """
    
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(pretrained=True).features
        self.slice = nn.Sequential(*list(vgg.children())[:16])
        for p in self.parameters():
            p.requires_grad = False
            
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))
        
        
    def forward(self, img):
        img = (img - self.mean) / self.std
        return self.slice(img)
    
def sample_path_coords(H, W, patch):
    top = random.randint(0, H - patch)
    left = random.randint(0, W - patch)
    rows = torch.arange(top, top + patch)
    cols = torch.arange(left, left + patch)
    yy, xx = torch.meshgrid(rows, cols, indexing='ij')
    coords = torch.stack([yy.reshape(-1), xx.reshape(-1)], dim=-1)
    return coords

def maxpool_feats(feat: torch.Tensor) -> torch.Tensor:
    # feat: [B,C,H,W] -> [B,C]
    return F.adaptive_max_pool2d(feat, 1).squeeze(-1).squeeze(-1)
    
def clamp_patch_top_left(y, x, H, W, patch):
    """패치 좌상단 좌표를 HxW 경계 안으로 클램프."""
    y0 = int(max(0, min(y - patch // 2, H - patch)))
    x0 = int(max(0, min(x - patch // 2, W - patch)))
    return y0, x0

def patch_coords_from_center(y, x, H, W, patch):
    """(y,x) 중심으로 patch_sz x patch_sz 패치의 모든 (yy,xx) 좌표 생성."""
    y0, x0 = clamp_patch_top_left(y, x, H, W, patch)
    rows = torch.arange(y0, y0 + patch)
    cols = torch.arange(x0, x0 + patch)
    yy, xx = torch.meshgrid(rows, cols, indexing='ij')
    return torch.stack([yy.reshape(-1), xx.reshape(-1)], dim=-1)  # (patch*patch, 2)

def sample_motion_patches(coords_np, H, W, patch, num_patches, device):
    """
    coords_np: np.ndarray (K,2) in (y,x), 모션 좌표 저장본
    num_patches: 뽑을 패치 개수
    return: torch.LongTensor [(num_patches*patch*patch), 2]
    """
    if coords_np.size == 0:
        return None  # 폴백 유도

    K = coords_np.shape[0]
    replace_flag = (K < num_patches)
    sel_idx = np.random.choice(K, size=[num_patches], replace=replace_flag)
    sel_centers = coords_np[sel_idx]

    all_coords = []
    for (yy, xx) in sel_centers:
        if 0 <= yy < H and 0 <= xx < W:
            pc = patch_coords_from_center(int(yy), int(xx), H, W, patch)
            all_coords.append(pc)
    if not all_coords:
        return None

    coords = torch.cat(all_coords, dim=0).to(device=device, dtype=torch.long)
    return coords  # (num_patches*patch*patch, 2)

def sample_random_patches(H, W, patch, num_patches, device):
    """무작위 패치 여러 개 샘플링."""
    all_coords = []
    for _ in range(num_patches):
        top = random.randint(0, max(0, H - patch))
        left = random.randint(0, max(0, W - patch))
        rows = torch.arange(top, top + patch, device=device)
        cols = torch.arange(left, left + patch, device=device)
        yy, xx = torch.meshgrid(rows, cols, indexing='ij')
        all_coords.append(torch.stack([yy.reshape(-1), xx.reshape(-1)], dim=-1))
    return torch.cat(all_coords, dim=0).long()

def sample_mixed_patches(coords_np, H, W, patch, num_patches, mix_ratio, device):
    """
    모션 기반 패치(f) + 랜덤 패치(1-f) 혼합.
    mix_ratio: 0.0~1.0 (모션 패치 비율)
    """
    n_motion = max(0, int(round(num_patches * mix_ratio)))
    n_rand   = max(0, num_patches - n_motion)

    coords_motion = sample_motion_patches(coords_np, H, W, patch, n_motion, device) if n_motion > 0 else None
    coords_rand   = sample_random_patches(H, W, patch, n_rand, device) if n_rand > 0 else None

    if coords_motion is None and n_motion > 0:
        # 모션 비었으면 전부 랜덤으로 대체
        return sample_random_patches(H, W, patch, num_patches, device)

    if coords_motion is None:
        return coords_rand
    if coords_rand is None:
        return coords_motion
    return torch.cat([coords_motion, coords_rand], dim=0)