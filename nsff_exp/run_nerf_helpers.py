import torch
torch.autograd.set_detect_anomaly(False)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
# TODO: remove this dependency
# from torchsearchsorted import searchsorted

# Misc
prob2weights = lambda x: x 

img2mse = lambda x, y : torch.mean((x - y) ** 2)
img2mae = lambda x, y : torch.mean(torch.abs(x - y))

mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)
tonemap = lambda x : (np.log(np.clip(x,0,1) * 50 + 1 ) / np.log(50 + 1)).astype(np.float32)


def im2tensor(image, imtype=np.uint8, cent=1., factor=1./2.):
    return torch.Tensor((image / factor - cent)
                        [:, :, :, np.newaxis].transpose((3, 2, 0, 1)))


# Positional encoding (section 5.1)
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0, input_dims=3):
    if i == -1:
        return nn.Identity(), 3
    
    embed_kwargs = {
                'include_input' : True,
                'input_dims' : input_dims,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }
    
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim


# Model
class NeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=True):
        """ 
        """
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)])
        
        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W//2)])

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])
        
        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W//2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)
        
        self.sf_linear = nn.Linear(W, 6)
        self.prob_linear = nn.Linear(W, 2)
        # self.blend_linear = nn.Linear(W // 2, 1)

    def forward(self, x):

        if self.use_viewdirs:
            input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        else:
            input_pts = x #torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)

        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        sf = nn.functional.tanh(self.sf_linear(h))
        prob = nn.functional.sigmoid(self.prob_linear(h))

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)
        
            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            
            # blend_w = nn.functional.sigmoid(self.blend_linear(h))

            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return torch.cat([outputs, sf, prob], dim=-1)



# Model
class Rigid_NeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=True):
        """ 
        """
        super(Rigid_NeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)])
        
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W//2)])

        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W//2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)
    
        self.w_linear = nn.Linear(W, 1)
        # h = F.relu(h)
        # blend_w = nn.functional.sigmoid(self.w_linear(h))

    def forward(self, x):

        if self.use_viewdirs:
            input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        else:
            input_pts = x #torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)

        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        v = nn.functional.sigmoid(self.w_linear(h))

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)
        
            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return torch.cat([outputs, v], -1)

# Ray helpers
def get_rays(H, W, focal, c2w):
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -torch.ones_like(i)], -1) # H,W,3
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs] # H,W,3
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    return rays_o, rays_d


# def get_rays_np(H, W, focal, c2w):
#     i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
#     dirs = np.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -np.ones_like(i)], -1)
#     # Rotate ray directions from camera frame to the world frame
#     rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
#     # Translate camera frame's origin to the world frame. It is the origin of all rays.
#     rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))
#     return rays_o, rays_d


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[...,2]) / rays_d[...,2]
    rays_o = rays_o + t[...,None] * rays_d
    
    # Projection
    o0 = -1./(W/(2.*focal)) * rays_o[...,0] / rays_o[...,2]
    o1 = -1./(H/(2.*focal)) * rays_o[...,1] / rays_o[...,2]
    o2 = 1. + 2. * near / rays_o[...,2]


    d0 = -1./(W/(2.*focal)) * (rays_d[...,0]/rays_d[...,2] - rays_o[...,0]/rays_o[...,2])
    d1 = -1./(H/(2.*focal)) * (rays_d[...,1]/rays_d[...,2] - rays_o[...,1]/rays_o[...,2])
    d2 = -2. * near / rays_o[...,2]
    
    rays_o = torch.stack([o0,o1,o2], -1)
    rays_d = torch.stack([d0,d1,d2], -1)
    
    return rays_o, rays_d


# # Hierarchical sampling (section 5.2)
# def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
#     # Get pdf
#     weights = weights + 1e-5 # prevent nans
#     pdf = weights / torch.sum(weights, -1, keepdim=True)
#     cdf = torch.cumsum(pdf, -1)
#     cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))

#     # Take uniform samples
#     if det:
#         u = torch.linspace(0., 1., steps=N_samples)
#         u = u.expand(list(cdf.shape[:-1]) + [N_samples])
#     else:
#         u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

#     # Pytest, overwrite u with numpy's fixed random numbers
#     if pytest:
#         np.random.seed(0)
#         new_shape = list(cdf.shape[:-1]) + [N_samples]
#         if det:
#             u = np.linspace(0., 1., N_samples)
#             u = np.broadcast_to(u, new_shape)
#         else:
#             u = np.random.rand(*new_shape)
#         u = torch.Tensor(u)

#     # Invert CDF
#     u = u.contiguous()
#     inds = searchsorted(cdf, u, side='right')
#     below = torch.max(torch.zeros_like(inds-1), inds-1)
#     above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
#     inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

#     # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
#     # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
#     matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
#     cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
#     bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

#     denom = (cdf_g[...,1]-cdf_g[...,0])
#     denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
#     t = (u-cdf_g[...,0])/denom
#     samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])

#     return samples



def compute_depth_loss(pred_depth, gt_depth):   
    # pred_depth_e = NDC2Euclidean(pred_depth_ndc)
    t_pred = torch.median(pred_depth)
    s_pred = torch.mean(torch.abs(pred_depth - t_pred))

    t_gt = torch.median(gt_depth)
    s_gt = torch.mean(torch.abs(gt_depth - t_gt))

    pred_depth_n = (pred_depth - t_pred)/s_pred
    gt_depth_n = (gt_depth - t_gt)/s_gt

    # return torch.mean(torch.abs(pred_depth_n - gt_depth_n))
    return torch.mean(torch.pow(pred_depth_n - gt_depth_n, 2))

def compute_depth_loss_o2(pred_depth, gt_depth):   
    # pred_depth_e = NDC2Euclidean(pred_depth, H, W, focal) # Since the near=0. far=1. the NDC to Euclidean doesn't need
    t_pred = torch.median(pred_depth)
    s_pred = torch.mean(torch.abs(pred_depth - t_pred))

    t_gt = torch.median(gt_depth)
    s_gt = torch.mean(torch.abs(gt_depth - t_gt))

    pred_depth_n = (pred_depth - t_pred)/s_pred
    gt_depth_n = (gt_depth - t_gt)/s_gt

    # return torch.mean(torch.abs(pred_depth_n - gt_depth_n))
    return torch.mean(torch.abs(pred_depth_n - gt_depth_n))

def compute_aligned_depth_loss(pred_depth, gt_depth):

    # Below code is from shape of motion
    # pred_depth = cast(torch.Tensor, rendered_all["depth"])
    # pred_disp = 1.0 / (pred_depth + 1e-5)
    # tgt_disp = 1.0 / (depths[..., None] + 1e-5)
    # depth_loss = masked_l1_loss(
    #     pred_disp,
    #     tgt_disp,
    #     mask=depth_masks,
    #     quantile=0.98,
    # )

    pred_disp = 1.0 / (pred_depth + 1e-5)
    tgt_disp = 1.0 / (gt_depth + 1e-5)

    return torch.mean(torch.pow(pred_disp - tgt_disp, 2))

def compute_mse(pred, gt, mask, dim=2):
    if dim == 1:
        mask_rep = torch.squeeze(mask)
    if dim == 2:
        mask_rep = mask.repeat(1, pred.size(-1))
    elif dim == 3:
        mask_rep = mask.repeat(1, 1, pred.size(-1))

    num_pix = torch.sum(mask_rep) + 1e-8
    return torch.sum( (pred - gt)**2 * mask_rep )/ num_pix

def compute_mae(pred, gt, mask, dim=2):
    if dim == 1:
        mask_rep = torch.squeeze(mask)
    if dim == 2:
        mask_rep = mask.repeat(1, pred.size(-1))
    elif dim == 3:
        mask_rep = mask.repeat(1, 1, pred.size(-1))

    num_pix = torch.sum(mask_rep) + 1e-8
    return torch.sum( torch.abs(pred - gt) * mask_rep )/ num_pix

def compute_mae_unit(pred, gt, mask, dim=2):
    if dim == 1:
        mask_rep = torch.squeeze(mask)
    if dim == 2:
        mask_rep = mask.repeat(1, pred.size(-1))
    elif dim == 3:
        mask_rep = mask.repeat(1, 1, pred.size(-1))

    num_pix = torch.sum(mask_rep) + 1e-8

    pred_scale = pred.pow(2).sum(2).sqrt()
    pred_scale = torch.unsqueeze(pred_scale, dim=-1).repeat(1,1,3)
    pred_unit = pred / pred_scale

    gt_scale = gt.pow(2).sum(2).sqrt()
    gt_scale = torch.unsqueeze(gt_scale, dim=-1).repeat(1,1,3)
    gt_unit = gt / gt_scale

    return torch.sum( torch.abs(pred_unit - gt_unit) * mask_rep )/ num_pix

def compute_mse_sat(pred, gt, mask, sat_mask, dim=2):
    if dim == 1:
        mask_rep = torch.squeeze(mask)
    if dim == 2:
        mask_rep = mask.repeat(1, pred.size(-1))
    elif dim == 3:
        mask_rep = mask.repeat(1, 1, pred.size(-1))

    num_pix = torch.sum(mask_rep) + 1e-8
    return torch.sum( (pred - gt)**2 * mask_rep * sat_mask )/ num_pix

def compute_mae_sat(pred, gt, mask, sat_mask, dim=2):
    if dim == 1:
        mask_rep = torch.squeeze(mask)
    if dim == 2:
        mask_rep = mask.repeat(1, pred.size(-1))
    elif dim == 3:
        mask_rep = mask.repeat(1, 1, pred.size(-1))

    num_pix = torch.sum(mask_rep) + 1e-8
    return torch.sum( torch.abs(pred - gt) * mask_rep * sat_mask)/ num_pix


# def compute_depth_loss_mask(pred_depth, gt_depth, mask):

#     mask = torch.squeeze(mask)

#     t_pred = torch.median(pred_depth)
#     s_pred = torch.mean(torch.abs(pred_depth - t_pred))

#     t_gt = torch.median(gt_depth)
#     s_gt = torch.mean(torch.abs(gt_depth - t_gt))

#     pred_depth_n = (pred_depth - t_pred)/s_pred
#     gt_depth_n = (gt_depth - t_gt)/s_gt

#     num_pixe = torch.sum(mask) + 1e-8

#     return torch.sum(torch.abs(pred_depth_n - gt_depth_n) * mask)/num_pixe


def normalize_depth(depth):
    # depth_sm = depth - torch.min(depth)
    return torch.clamp(depth/percentile(depth, 97), 0., 1.)


def percentile(t, q):
    """
    Return the ``q``-th percentile of the flattened input tensor's data.
    
    CAUTION:
     * Needs PyTorch >= 1.1.0, as ``torch.kthvalue()`` is used.
     * Values are not interpolated, which corresponds to
       ``numpy.percentile(..., interpolation="nearest")``.
       
    :param t: Input tensor.
    :param q: Percentile to compute, which must be between 0 and 100 inclusive.
    :return: Resulting value (scalar).
    """
    # Note that ``kthvalue()`` works one-based, i.e. the first sorted value
    # indeed corresponds to k=1, not k=0! Use float(q) instead of q directly,
    # so that ``round()`` returns an integer, even if q is a np.float32.
    k = 1 + round(.01 * float(q) * (t.numel() - 1))
    result = t.view(-1).kthvalue(k).values.item()
    return result


def make_color_wheel():
  """
  Generate color wheel according Middlebury color code
  :return: Color wheel
  """
  RY = 15
  YG = 6
  GC = 4
  CB = 11
  BM = 13
  MR = 6

  ncols = RY + YG + GC + CB + BM + MR

  colorwheel = np.zeros([ncols, 3])

  col = 0

  # RY
  colorwheel[0:RY, 0] = 255
  colorwheel[0:RY, 1] = np.transpose(np.floor(255*np.arange(0, RY) / RY))
  col += RY

  # YG
  colorwheel[col:col+YG, 0] = 255 - np.transpose(np.floor(255*np.arange(0, YG) / YG))
  colorwheel[col:col+YG, 1] = 255
  col += YG

  # GC
  colorwheel[col:col+GC, 1] = 255
  colorwheel[col:col+GC, 2] = np.transpose(np.floor(255*np.arange(0, GC) / GC))
  col += GC

  # CB
  colorwheel[col:col+CB, 1] = 255 - np.transpose(np.floor(255*np.arange(0, CB) / CB))
  colorwheel[col:col+CB, 2] = 255
  col += CB

  # BM
  colorwheel[col:col+BM, 2] = 255
  colorwheel[col:col+BM, 0] = np.transpose(np.floor(255*np.arange(0, BM) / BM))
  col += + BM

  # MR
  colorwheel[col:col+MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
  colorwheel[col:col+MR, 0] = 255

  return colorwheel


def compute_color(u, v):
  """
  compute optical flow color map
  :param u: optical flow horizontal map
  :param v: optical flow vertical map
  :return: optical flow in color code
  """
  [h, w] = u.shape
  img = np.zeros([h, w, 3])
  nanIdx = np.isnan(u) | np.isnan(v)
  u[nanIdx] = 0
  v[nanIdx] = 0

  colorwheel = make_color_wheel()
  ncols = np.size(colorwheel, 0)

  rad = np.sqrt(u**2+v**2)

  a = np.arctan2(-v, -u) / np.pi

  fk = (a+1) / 2 * (ncols - 1) + 1

  k0 = np.floor(fk).astype(int)

  k1 = k0 + 1
  k1[k1 == ncols+1] = 1
  f = fk - k0

  for i in range(0, np.size(colorwheel,1)):
    tmp = colorwheel[:, i]
    col0 = tmp[k0-1] / 255
    col1 = tmp[k1-1] / 255
    col = (1-f) * col0 + f * col1

    idx = rad <= 1
    col[idx] = 1-rad[idx]*(1-col[idx])
    notidx = np.logical_not(idx)

    col[notidx] *= 0.75
    img[:, :, i] = np.uint8(np.floor(255 * col*(1-nanIdx)))

  return img


def flow_to_image(flow, display=False):
  """
  Convert flow into middlebury color code image
  :param flow: optical flow map
  :return: optical flow image in middlebury color
  """
  UNKNOWN_FLOW_THRESH = 100
  u = flow[:, :, 0]
  v = flow[:, :, 1]

  maxu = -999.
  maxv = -999.
  minu = 999.
  minv = 999.

  idxUnknow = (abs(u) > UNKNOWN_FLOW_THRESH) | (abs(v) > UNKNOWN_FLOW_THRESH)
  u[idxUnknow] = 0
  v[idxUnknow] = 0

  maxu = max(maxu, np.max(u))
  minu = min(minu, np.min(u))

  maxv = max(maxv, np.max(v))
  minv = min(minv, np.min(v))

  # sqrt_rad = u**2 + v**2
  rad = np.sqrt(u**2 + v**2)

  maxrad = max(-1, np.max(rad))

  if display:
    print("max flow: %.4f\nflow range:\nu = %.3f .. %.3f\nv = %.3f .. %.3f" % (maxrad, minu,maxu, minv, maxv))

  u = u/(maxrad + np.finfo(float).eps)
  v = v/(maxrad + np.finfo(float).eps)

  img = compute_color(u, v)

  idx = np.repeat(idxUnknow[:, :, np.newaxis], 3, axis=2)
  img[idx] = 0

  return np.uint8(img)

# NOTE: WE DO IN COLMAP/OPENCV FORMAT, BUT INPUT IS OPENGL FORMAT!!!!!1
def perspective_projection(pts_3d, h, w, f):
    pts_2d = torch.cat([pts_3d[..., 0:1] * f/-pts_3d[..., 2:3] + w/2., 
                        -pts_3d[..., 1:2] * f/-pts_3d[..., 2:3] + h/2.], dim=-1)

    return pts_2d    


def se3_transform_points(pts_ref, raw_rot_ref2prev, raw_trans_ref2prev):
    pts_prev = torch.squeeze(torch.matmul(raw_rot_ref2prev, pts_ref[..., :3].unsqueeze(-1)) + raw_trans_ref2prev)
    return pts_prev


def NDC2Euclidean(xyz_ndc, H, W, f):
    z_e = 2./ (xyz_ndc[..., 2:3] - 1. + 1e-6)
    x_e = - xyz_ndc[..., 0:1] * z_e * W/ (2. * f)
    y_e = - xyz_ndc[..., 1:2] * z_e * H/ (2. * f)

    xyz_e = torch.cat([x_e, y_e, z_e], -1)
 
    return xyz_e

import sys


def projection_from_ndc(c2w, H, W, f, weights_ref, raw_pts, n_dim=1):
    R_w2c = c2w[:3, :3].transpose(0, 1)
    t_w2c = -torch.matmul(R_w2c, c2w[:3, 3:])

    pts_3d = torch.sum(weights_ref[...,None] * raw_pts, -2)  # [N_rays, 3]

    pts_3d_e_world = NDC2Euclidean(pts_3d, H, W, f)

    if n_dim == 1:
        pts_3d_e_local = se3_transform_points(pts_3d_e_world, 
                                              R_w2c.unsqueeze(0), 
                                              t_w2c.unsqueeze(0))
    else:
        pts_3d_e_local = se3_transform_points(pts_3d_e_world, 
                                              R_w2c.unsqueeze(0).unsqueeze(0), 
                                              t_w2c.unsqueeze(0).unsqueeze(0))

    pts_2d = perspective_projection(pts_3d_e_local, H, W, f)

    return pts_2d

def projection_from_ndc_multi_camera(c2w, H, W, f_ref, f_prev, weights_ref, raw_pts, n_dim=1):
    R_w2c = c2w[:3, :3].transpose(0, 1)
    t_w2c = -torch.matmul(R_w2c, c2w[:3, 3:])

    pts_3d = torch.sum(weights_ref[...,None] * raw_pts, -2)  # [N_rays, 3]

    pts_3d_e_world = NDC2Euclidean(pts_3d, H, W, f_ref)

    if n_dim == 1:
        pts_3d_e_local = se3_transform_points(pts_3d_e_world, 
                                              R_w2c.unsqueeze(0), 
                                              t_w2c.unsqueeze(0))
    else:
        pts_3d_e_local = se3_transform_points(pts_3d_e_world, 
                                              R_w2c.unsqueeze(0).unsqueeze(0), 
                                              t_w2c.unsqueeze(0).unsqueeze(0))

    pts_2d = perspective_projection(pts_3d_e_local, H, W, f_prev)

    return pts_2d

def compute_optical_flow(pose_post, pose_ref, pose_prev, H, W, focal, ret, n_dim=1):
    pts_2d_post = projection_from_ndc(pose_post, H, W, focal, 
                                      ret['weights_ref_dy'], 
                                      ret['raw_pts_post'], 
                                      n_dim)
    pts_2d_prev = projection_from_ndc(pose_prev, H, W, focal, 
                                      ret['weights_ref_dy'], 
                                      ret['raw_pts_prev'], 
                                      n_dim)

    return pts_2d_post, pts_2d_prev

def compute_optical_flow_i2(pose_postpost, pose_ref, pose_prevprev, H, W, focal, ret, n_dim=1):
    pts_2d_postpost = projection_from_ndc(pose_postpost, H, W, focal, 
                                      ret['weights_ref_dy'], 
                                      ret['raw_pts_post'], 
                                      n_dim)
    pts_2d_prevprev = projection_from_ndc(pose_prevprev, H, W, focal, 
                                      ret['weights_ref_dy'], 
                                      ret['raw_pts_prev'], 
                                      n_dim)

    return pts_2d_postpost, pts_2d_prevprev

def compute_optical_flow_multi_camera(pose_post, pose_ref, pose_prev, H, W, focal_post, focal_ref, focal_prev, ret, n_dim=1):
    pts_2d_post = projection_from_ndc_multi_camera(pose_post, H, W, focal_ref, focal_post, 
                                      ret['weights_ref_dy'], 
                                      ret['raw_pts_post'], 
                                      n_dim)
    pts_2d_prev = projection_from_ndc_multi_camera(pose_prev, H, W, focal_ref, focal_prev, 
                                      ret['weights_ref_dy'], 
                                      ret['raw_pts_prev'], 
                                      n_dim)

    return pts_2d_post, pts_2d_prev


def read_optical_flow(basedir, img_i, start_frame, fwd):
    import os
    flow_dir = os.path.join(basedir, 'flow_i1')

    if fwd:
      fwd_flow_path = os.path.join(flow_dir, 
                                  '%05d_fwd.npz'%(start_frame + img_i))
      fwd_data = np.load(fwd_flow_path)#, (w, h))
      fwd_flow, fwd_mask = fwd_data['flow'], fwd_data['mask']
      fwd_mask = np.float32(fwd_mask)  
      
      return fwd_flow, fwd_mask
    else:

      bwd_flow_path = os.path.join(flow_dir, 
                                  '%05d_bwd.npz'%(start_frame + img_i))

      bwd_data = np.load(bwd_flow_path)#, (w, h))
      bwd_flow, bwd_mask = bwd_data['flow'], bwd_data['mask']
      bwd_mask = np.float32(bwd_mask)

      return bwd_flow, bwd_mask
  
def read_optical_flow(basedir, img_i, start_frame, fwd, step):
    import os
    flow_dir = os.path.join(basedir, f'flow_i{step}')
    
    start_frame = start_frame // step
    
    # import pdb; pdb.set_trace()

    if fwd:
      fwd_flow_path = os.path.join(flow_dir, 
                                  '%05d_fwd.npz'%(start_frame + img_i))
      fwd_data = np.load(fwd_flow_path)#, (w, h))
      fwd_flow, fwd_mask = fwd_data['flow'], fwd_data['mask']
      fwd_mask = np.float32(fwd_mask)  
      
      return fwd_flow, fwd_mask
    else:

      bwd_flow_path = os.path.join(flow_dir, 
                                  '%05d_bwd.npz'%(start_frame + img_i))

      bwd_data = np.load(bwd_flow_path)#, (w, h))
      bwd_flow, bwd_mask = bwd_data['flow'], bwd_data['mask']
      bwd_mask = np.float32(bwd_mask)

      return bwd_flow, bwd_mask
  
def read_semantic_flow_(basedir, img_i, fwd):
    import os
    flow_dir = os.path.join(basedir, 'semantic_flow_i1')

    if fwd:
      fwd_flow_path = os.path.join(flow_dir, 
                                  '%05d_fwd.npz'%(img_i))
      fwd_data = np.load(fwd_flow_path)#, (w, h))
      fwd_flow, segm_mask = fwd_data['flow'], fwd_data['mask']
      segm_mask = np.float32(segm_mask)  
      
      return fwd_flow, segm_mask
    else:

      bwd_flow_path = os.path.join(flow_dir, 
                                  '%05d_bwd.npz'%(img_i))

      bwd_data = np.load(bwd_flow_path)#, (w, h))
      bwd_flow, segm_mask = bwd_data['flow'], bwd_data['mask']
      segm_mask = np.float32(segm_mask)

      return bwd_flow, segm_mask
    
def read_semantic_flow(basedir, img_i, fwd, start_frame=0, step=1):
    import os
    flow_dir = os.path.join(basedir, f'semantic_flow_i{step}')

    # start_frame = start_frame // step

    if fwd:
      fwd_flow_path = os.path.join(flow_dir, 
                                  '%05d_fwd.npz'%(start_frame + img_i))
      fwd_data = np.load(fwd_flow_path)#, (w, h))
      fwd_flow, segm_mask = fwd_data['flow'], fwd_data['mask']
      segm_mask = np.float32(segm_mask)  
      
      return fwd_flow, segm_mask
    else:

      bwd_flow_path = os.path.join(flow_dir, 
                                  '%05d_bwd.npz'%(start_frame + img_i))

      bwd_data = np.load(bwd_flow_path)#, (w, h))
      bwd_flow, segm_mask = bwd_data['flow'], bwd_data['mask']
      segm_mask = np.float32(segm_mask)

      return bwd_flow, segm_mask
    
def read_semantic_flow_no_offset(basedir, img_i, fwd, start_frame=0, step=1):
    import os
    flow_dir = os.path.join(basedir, f'semantic_flow_i{step}')

    # start_frame = start_frame // step

    if fwd:
      fwd_flow_path = os.path.join(flow_dir, 
                                  '%05d_fwd.npz'%(img_i))
      fwd_data = np.load(fwd_flow_path)#, (w, h))
      fwd_flow, segm_mask = fwd_data['flow'], fwd_data['mask']
      segm_mask = np.float32(segm_mask)  
      
      return fwd_flow, segm_mask
    else:

      bwd_flow_path = os.path.join(flow_dir, 
                                  '%05d_bwd.npz'%(img_i))

      bwd_data = np.load(bwd_flow_path)#, (w, h))
      bwd_flow, segm_mask = bwd_data['flow'], bwd_data['mask']
      segm_mask = np.float32(segm_mask)

      return bwd_flow, segm_mask


def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow_new = flow.copy()
    flow_new[:,:,0] += np.arange(w)
    flow_new[:,:,1] += np.arange(h)[:,np.newaxis]

    res = cv2.remap(img, flow_new, None, 
                    cv2.INTER_CUBIC, 
                    borderMode=cv2.BORDER_CONSTANT)
    return res

def compute_sf_sm_loss(pts_1_ndc, pts_2_ndc, H, W, f):
    # sigma = 2.
    n = pts_1_ndc.shape[1]

    pts_1_ndc_close = pts_1_ndc[..., :int(n * 0.95), :]
    pts_2_ndc_close = pts_2_ndc[..., :int(n * 0.95), :]

    pts_3d_1_world = NDC2Euclidean(pts_1_ndc_close, H, W, f)
    pts_3d_2_world = NDC2Euclidean(pts_2_ndc_close, H, W, f)
        
    # dist = torch.norm(pts_3d_1_world[..., :-1, :] - pts_3d_1_world[..., 1:, :], 
                      # dim=-1, keepdim=True)
    # weights = torch.exp(-dist * sigma).detach()

    # scene flow 
    scene_flow_world = pts_3d_1_world - pts_3d_2_world

    return torch.mean(torch.abs(scene_flow_world[..., :-1, :] - scene_flow_world[..., 1:, :]))

def compute_sf_sm_loss_multi_camera(pts_1_ndc, pts_2_ndc, H, W, f1, f2):
    # sigma = 2.
    n = pts_1_ndc.shape[1]

    pts_1_ndc_close = pts_1_ndc[..., :int(n * 0.95), :]
    pts_2_ndc_close = pts_2_ndc[..., :int(n * 0.95), :]

    pts_3d_1_world = NDC2Euclidean(pts_1_ndc_close, H, W, f1)
    pts_3d_2_world = NDC2Euclidean(pts_2_ndc_close, H, W, f1)
        
    # dist = torch.norm(pts_3d_1_world[..., :-1, :] - pts_3d_1_world[..., 1:, :], 
                      # dim=-1, keepdim=True)
    # weights = torch.exp(-dist * sigma).detach()

    # scene flow 
    scene_flow_world = pts_3d_1_world - pts_3d_2_world

    return torch.mean(torch.abs(scene_flow_world[..., :-1, :] - scene_flow_world[..., 1:, :]))

# Least kinetic motion prior
def compute_sf_lke_loss(pts_ref_ndc, pts_post_ndc, pts_prev_ndc, H, W, f):
    n = pts_ref_ndc.shape[1]

    pts_ref_ndc_close = pts_ref_ndc[..., :int(n * 0.9), :]
    pts_post_ndc_close = pts_post_ndc[..., :int(n * 0.9), :]
    pts_prev_ndc_close = pts_prev_ndc[..., :int(n * 0.9), :]

    pts_3d_ref_world = NDC2Euclidean(pts_ref_ndc_close, 
                                     H, W, f)
    pts_3d_post_world = NDC2Euclidean(pts_post_ndc_close, 
                                     H, W, f)
    pts_3d_prev_world = NDC2Euclidean(pts_prev_ndc_close, 
                                     H, W, f)
    
    # scene flow 
    scene_flow_w_ref2post = pts_3d_post_world - pts_3d_ref_world
    scene_flow_w_prev2ref = pts_3d_ref_world - pts_3d_prev_world

    return 0.5 * torch.mean((scene_flow_w_ref2post - scene_flow_w_prev2ref) ** 2)

# Least kinetic motion prior
def compute_sf_lke_loss_multi_camera(pts_ref_ndc, pts_post_ndc, pts_prev_ndc, H, W, f, f_post, f_prev):
    n = pts_ref_ndc.shape[1]

    pts_ref_ndc_close = pts_ref_ndc[..., :int(n * 0.9), :]
    pts_post_ndc_close = pts_post_ndc[..., :int(n * 0.9), :]
    pts_prev_ndc_close = pts_prev_ndc[..., :int(n * 0.9), :]

    pts_3d_ref_world = NDC2Euclidean(pts_ref_ndc_close, 
                                     H, W, f)
    pts_3d_post_world = NDC2Euclidean(pts_post_ndc_close, 
                                     H, W, f_post)
    pts_3d_prev_world = NDC2Euclidean(pts_prev_ndc_close, 
                                     H, W, f_prev)
    
    # scene flow 
    scene_flow_w_ref2post = pts_3d_post_world - pts_3d_ref_world
    scene_flow_w_prev2ref = pts_3d_ref_world - pts_3d_prev_world

    return 0.5 * torch.mean((scene_flow_w_ref2post - scene_flow_w_prev2ref) ** 2)

def saturation_mask(color, threshold_low=0.15,threshold_high=0.9):
    mask = torch.zeros_like(color)
    mask[(color<=threshold_high) & (color>=threshold_low)] = 1
    mask[(color<threshold_low)] = torch.pow((color[(color<=threshold_low)]+threshold_low)/(2*threshold_low),2)
    mask[(color>threshold_high)] = torch.pow((color[(color>threshold_high)]-2+threshold_high)/(2*(1-threshold_high)),2)

    return mask


def saturation_mask_3(color, threshold_low=0.15, threshold_high=0.9):
    mask = torch.zeros_like(color)
    mask[(color <= threshold_high) & (color >= threshold_low)] = 1
    
    # threshold_low 근처에서 연속적으로 1/9까지 감소
    mask[(color < threshold_low)] = (1/9) + (8/9) * torch.pow((color[(color < threshold_low)] / threshold_low), 2)
    
    # threshold_high 근처에서 연속적으로 1/9까지 감소
    mask[(color > threshold_high)] = (1/9) + (8/9) * torch.pow((1 - color[(color > threshold_high)]) / (1 - threshold_high), 2)
    
    return mask

def saturation_mask_0(color, threshold_low=0.15, threshold_high=0.9):
    mask = torch.zeros_like(color)
    mask[(color <= threshold_high) & (color >= threshold_low)] = 1
    
    # threshold_low 근처에서 연속적으로 0까지 감소
    mask[(color < threshold_low)] = torch.pow((color[(color < threshold_low)] / threshold_low), 2)
    
    # threshold_high 근처에서 연속적으로 0까지 감소
    mask[(color > threshold_high)] = torch.pow((1 - color[(color > threshold_high)]) / (1 - threshold_high), 2)
    
    return mask

def saturation_mask_0_low(color, threshold_low=0.15, threshold_high=0.9):
    mask = torch.zeros_like(color)
    mask[(color >= threshold_low)] = 1
    
    # threshold_low 근처에서 연속적으로 0까지 감소
    mask[(color < threshold_low)] = torch.pow((color[(color < threshold_low)] / threshold_low), 2)
    
    # # threshold_high 근처에서 연속적으로 0까지 감소
    # mask[(color > threshold_high)] = torch.pow((1 - color[(color > threshold_high)]) / (1 - threshold_high), 2)
    
    return mask

def saturation_mask_0_high(color, threshold_low=0.15, threshold_high=0.9):
    mask = torch.zeros_like(color)
    mask[(color <= threshold_high)] = 1
    
    # # threshold_low 근처에서 연속적으로 0까지 감소
    # mask[(color < threshold_low)] = torch.pow((color[(color < threshold_low)] / threshold_low), 2)
    # threshold_high 근처에서 연속적으로 0까지 감소
    mask[(color > threshold_high)] = torch.pow((1 - color[(color > threshold_high)]) / (1 - threshold_high), 2)
    
    return mask

def grad_on(m):
    return any(p.requires_grad for p in m.parameters()) if (m is not None) else None

def reproject_mask_to_view__(mask_ref, depth_ref, c2w_ref, c2w_syn, H, W, focal):
    """
    mask_ref: [H, W] float/bool
    depth_ref: [H, W] in meters (or consistent scale)
    c2w_ref: [3,4] (camera-to-world of reference)
    c2w_syn: [3,4] (camera-to-world of synthetic view)
    focal: scalar (fx=fy)
    return: mask_syn [H, W] in {0,1}
    """
    device = depth_ref.device
    # intrinsics
    cx = (W - 1) * 0.5
    cy = (H - 1) * 0.5

    # pixel grid
    ys, xs = torch.meshgrid(torch.arange(H, device=device),
                            torch.arange(W, device=device),
                            indexing='ij')
    zs = depth_ref.clamp(min=1e-6)
    Xc = torch.stack([
        (xs - cx) / focal * zs,
        (ys - cy) / focal * zs,
        zs
    ], dim=-1)  # [H,W,3] in ref cam

    # cam-to-world (3x4) -> world-to-cam (syn)
    R_ref, t_ref = c2w_ref[:,:3], c2w_ref[:,3:]
    R_syn, t_syn = c2w_syn[:,:3], c2w_syn[:,3:]

    # to world
    Xw = (R_ref @ Xc.reshape(-1,3).T + t_ref).T  # [HW,3]
    # to synthetic cam
    w2c_syn_R = R_syn.T
    w2c_syn_t = - (R_syn.T @ t_syn)  # [3,1]
    Xs = (w2c_syn_R @ Xw.T + w2c_syn_t).T  # [HW,3]

    Z = Xs[:,2].clamp(min=1e-6)
    us = (Xs[:,0] / Z) * focal + cx
    vs = (Xs[:,1] / Z) * focal + cy

    # valid proj
    valid = (Z > 0) & (us >= 0) & (us <= W-1) & (vs >= 0) & (vs <= H-1)

    # Z-buffer: keep nearest z per (u,v) integer bin
    u0 = us[valid].round().long()
    v0 = vs[valid].round().long()
    z0 = Z[valid]
    m0 = mask_ref.reshape(-1)[valid].float()

    # initialize with +inf
    zbuf = torch.full((H,W), float('inf'), device=device)
    out  = torch.zeros((H,W), dtype=torch.float32, device=device)

    # for duplicates, keep nearest
    # scatter: create a linear index
    lin_idx = v0 * W + u0
    # get nearest depth per pixel
    # sort by depth asc
    sort_idx = torch.argsort(z0)
    lin_idx, z0, m0 = lin_idx[sort_idx], z0[sort_idx], m0[sort_idx]

    is_first = torch.ones_like(lin_idx, dtype=torch.bool)
    is_first[1:] = lin_idx[1:] != lin_idx[:-1]

    keep = is_first  # 가장 가까운 샘플만 유지

    # (선택) z-buffer 저장하고 싶다면
    zbuf_flat = zbuf.view(-1)
    out_flat  = out.view(-1)

    zbuf_flat[lin_idx[keep]] = z0[keep]
    out_flat[lin_idx[keep]]  = m0[keep]

    # 이진화
    out = (out > 0.5).float()
    # # only first occurrence per pixel wins (nearest)
    # unique_lin, first_pos = torch.unique_consecutive(lin_idx, return_counts=False, return_inverse=False, return_counts=None, dim=0)
    # # we can mark by writing once
    # zbuf.view(-1)[lin_idx] = torch.minimum(zbuf.view(-1)[lin_idx], z0)
    # out.view(-1)[lin_idx]  = m0  # nearest already sorted

    # # binarize
    # out = (out > 0.5).float()
    return out

def reproject_mask_to_view(mask_ref, depth_ref, c2w_ref, c2w_syn, H, W, focal, eps=1e-6):
    """
    mask_ref: [H,W] {0/1} or float/bool (ref view)
    depth_ref: [H,W] ref-view depth
    c2w_ref, c2w_syn: [3,4]
    return: [H,W] {0,1}
    """
    device = depth_ref.device
    mask_ref = mask_ref.to(device)
    c2w_ref = c2w_ref.to(device)
    c2w_syn = c2w_syn.to(device)

    # intrinsics
    cx = (W - 1) * 0.5
    cy = (H - 1) * 0.5

    # pixel grid
    ys, xs = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing='ij'
    )

    # --- ref cam 3D (from ref depth) ---
    zs = depth_ref.clamp(min=eps)
    Xc = torch.stack([
        (xs - cx) / focal * zs,
        (ys - cy) / focal * zs,
        zs
    ], dim=-1)                              # [H,W,3]

    # --- cam->world (ref), world->cam (syn) ---
    R_ref, t_ref = c2w_ref[:, :3], c2w_ref[:, 3:].reshape(3,1)
    R_syn, t_syn = c2w_syn[:, :3], c2w_syn[:, 3:].reshape(3,1)

    Xw = (R_ref @ Xc.reshape(-1,3).T + t_ref).T          # [HW,3]
    w2c_R = R_syn.T
    w2c_t = -(w2c_R @ t_syn)                              # [3,1]
    Xs = (w2c_R @ Xw.T + w2c_t).T                         # [HW,3]

    Z = Xs[:, 2].clamp(min=eps)
    us = (Xs[:, 0] / Z) * focal + cx
    vs = (Xs[:, 1] / Z) * focal + cy

    # --- valid & mask>0만 후보로 사용 ---
    in_img = (us >= 0) & (us <= W - 1) & (vs >= 0) & (vs <= H - 1)
    m_flat = (mask_ref.reshape(-1) > 0.5)
    cand = in_img & (Z > 0) & m_flat
    if not torch.any(cand):
        return torch.zeros((H, W), device=device, dtype=torch.float32)

    u0 = us[cand].round().long()
    v0 = vs[cand].round().long()
    z0 = Z[cand]

    lin = v0 * W + u0                          # target linear indices

    # --- 픽셀별 최소 Z 집계 (정확한 Z-buffer) ---
    HW = H * W
    zmin = torch.full((HW,), float('inf'), device=device)
    # PyTorch 2.x
    zmin.scatter_reduce_(0, lin, z0, reduce='amin', include_self=True)

    # keep if z0 is (almost) the pixel's min z
    keep = z0 <= (zmin[lin] + 1e-6)

    out = torch.zeros((HW,), device=device, dtype=torch.float32)
    out[lin[keep]] = 1.0
    out = out.view(H, W)

    return out

@torch.no_grad()
def project_mask_points(
    mask_ref: torch.Tensor,    # [H,W] 0/1, bool, or float
    depth_ref: torch.Tensor,   # [H,W] ref depth (same units as poses)
    c2w_ref: torch.Tensor,     # [3,4]
    c2w_syn: torch.Tensor,     # [3,4]
    H: int, W: int, focal: float,
    eps: float = 1e-6,
    return_sparse_mask: bool = False,
):
    """
    ref의 마스크 픽셀만 골라 3D→syn뷰 투영 좌표 (u,v) 계산.
    - 출력은 '유효한' 점들만(이미지 안 & Z>0) 반환.
    - return_sparse_mask=True면, syn 뷰에 라운드된 위치에 1 찍은 희소 마스크도 함께 반환.

    Returns:
        us:   [N] syn u 좌표 (float)
        vs:   [N] syn v 좌표 (float)
        idxs: [N,2] 원본 ref 픽셀 (y,x) 인덱스 (int64)
        (opt) mask_syn_sparse: [H,W] float32 {0,1}  # 요청 시
    """
    device = depth_ref.device
    mask_ref = mask_ref.to(device)
    c2w_ref = c2w_ref.to(device)
    c2w_syn = c2w_syn.to(device)

    # 1) ref에서 마스크가 켜진 픽셀 좌표만 추출
    if mask_ref.dtype != torch.bool:
        m = mask_ref > 0.5
    else:
        m = mask_ref
    ys, xs = torch.nonzero(m, as_tuple=True)    # [N], [N]
    if ys.numel() == 0:
        if return_sparse_mask:
            return (torch.empty(0, device=device), torch.empty(0, device=device),
                    torch.empty((0,2), dtype=torch.long, device=device),
                    torch.zeros((H,W), dtype=torch.float32, device=device))
        else:
            return (torch.empty(0, device=device), torch.empty(0, device=device),
                    torch.empty((0,2), dtype=torch.long, device=device))

    # 2) 해당 픽셀들의 깊이
    z = depth_ref[ys, xs].clamp(min=eps)        # [N]

    # 3) ref 카메라좌표계 3D (pinhole: x = (u-cx)/f * z, y = (v-cy)/f * z)
    cx = (W - 1) * 0.5
    cy = (H - 1) * 0.5
    Xc = torch.stack([
        (xs.float() - cx) / float(focal) * z,
        (ys.float() - cy) / float(focal) * z,
        z
    ], dim=-1)                                  # [N,3]

    # 4) ref c2w → 월드
    R_ref = c2w_ref[:, :3]                      # [3,3]
    t_ref = c2w_ref[:, 3]                       # [3]
    Xw = (Xc @ R_ref.T) + t_ref                 # [N,3]

    # 5) syn w2c
    R_syn = c2w_syn[:, :3]
    t_syn = c2w_syn[:, 3]
    w2c_R = R_syn.T
    w2c_t = -(w2c_R @ t_syn)                    # [3]
    Xs = (Xw @ w2c_R.T) + w2c_t                 # [N,3]

    # 6) 투영 (u,v), 유효성 필터 (Z>0, 이미지 안)
    Zs = Xs[:, 2].clamp(min=eps)                # [N]
    us = (Xs[:, 0] / Zs) * float(focal) + cx
    vs = (Xs[:, 1] / Zs) * float(focal) + cy

    in_img = (us >= 0) & (us <= (W - 1)) & (vs >= 0) & (vs <= (H - 1))
    in_front = Zs > 0
    valid = in_img & in_front
    us, vs = us[valid], vs[valid]
    idxs = torch.stack([ys[valid], xs[valid]], dim=-1)  # [Nv,2] (y,x)

    if not return_sparse_mask:
        return us, vs, idxs

    # 7) 희소 마스크(라운드한 픽셀에만 1 찍기; 겹치면 그냥 1)
    u0 = us.round().long()
    v0 = vs.round().long()
    mask_syn_sparse = torch.zeros((H, W), dtype=torch.float32, device=device)
    mask_syn_sparse[v0, u0] = 1.0

    return us, vs, idxs, mask_syn_sparse


@torch.no_grad()
def project_mask_points_v2(
    mask_ref: torch.Tensor,    # [H,W] 0/1, bool, or float
    depth_ref: torch.Tensor,   # [H,W]  cam-Z or NDC t (depth_mode로 지정)
    c2w_ref: torch.Tensor,     # [3,4]
    c2w_syn: torch.Tensor,     # [3,4]
    H: int, W: int, focal: float,
    eps: float = 1e-6,
    return_sparse_mask: bool = False,
    depth_mode: str = 'ndc_t', # 'cam_z' | 'ndc_t'
    near: float = 1.0,         # depth_mode='ndc_t'일 때 사용
):
    """
    ref의 마스크 픽셀만 골라 3D→syn뷰 투영 좌표 (u,v) 계산.
    depth_mode:
      - 'cam_z' : depth_ref가 카메라 z-깊이 (미터 등)일 때
      - 'ndc_t' : depth_ref가 NDC 정규화 깊이 t∈[0,1)일 때 (z = near/(1-t))
    """
    device = depth_ref.device
    mask_ref = mask_ref.to(device)
    c2w_ref = c2w_ref.to(device)
    c2w_syn = c2w_syn.to(device)

    # 1) ref에서 마스크가 켜진 픽셀 좌표만 추출
    m = (mask_ref > 0.5) if mask_ref.dtype != torch.bool else mask_ref
    ys, xs = torch.nonzero(m, as_tuple=True)    # [N], [N]
    if ys.numel() == 0:
        if return_sparse_mask:
            return (torch.empty(0, device=device), torch.empty(0, device=device),
                    torch.empty((0,2), dtype=torch.long, device=device),
                    torch.zeros((H,W), dtype=torch.float32, device=device))
        else:
            return (torch.empty(0, device=device), torch.empty(0, device=device),
                    torch.empty((0,2), dtype=torch.long, device=device))

    # 2) 해당 픽셀들의 깊이 -> cam-Z로 변환
    d = depth_ref[ys, xs].float()  # [N]
    if depth_mode == 'cam_z':
        z = d.clamp(min=eps)
    elif depth_mode == 'ndc_t':
        # t∈[0,1) → z = near / (1 - t)
        # t가 1에 너무 가까우면 폭주하므로 안정화
        t = torch.clamp(d, 0.0, 1.0 - 1e-6)
        z = (near / (1.0 - t)).clamp(min=eps)
    else:
        raise ValueError("depth_mode must be 'cam_z' or 'ndc_t'")

    # 3) ref 카메라좌표계 3D
    cx = (W - 1) * 0.5
    cy = (H - 1) * 0.5
    f  = float(focal)
    Xc = torch.stack([
        (xs.float() - cx) / f * z,
        (ys.float() - cy) / f * z,
        z
    ], dim=-1)                                  # [N,3]

    # 4) ref c2w → 월드
    R_ref = c2w_ref[:, :3]                      # [3,3]
    t_ref = c2w_ref[:, 3]                       # [3]
    Xw = (Xc @ R_ref.T) + t_ref                 # [N,3]

    # 5) syn w2c
    R_syn = c2w_syn[:, :3]
    t_syn = c2w_syn[:, 3]
    w2c_R = R_syn.T
    w2c_t = -(w2c_R @ t_syn)                    # [3]
    Xs = (Xw @ w2c_R.T) + w2c_t                 # [N,3]

    # 6) 투영 (u,v), 유효성 필터 (Z>0, 이미지 안)
    Zs = Xs[:, 2].clamp(min=eps)                # [N]
    us = (Xs[:, 0] / Zs) * f + cx
    vs = (Xs[:, 1] / Zs) * f + cy

    in_img = (us >= 0) & (us <= (W - 1)) & (vs >= 0) & (vs <= (H - 1))
    in_front = Zs > 0
    valid = in_img & in_front

    us, vs = us[valid], vs[valid]
    idxs = torch.stack([ys[valid], xs[valid]], dim=-1)  # [Nv,2] (y,x)

    if not return_sparse_mask:
        return us, vs, idxs

    # 7) 희소 마스크(라운드 + 클램프로 안전하게)
    u0 = us.round().clamp_(0, W-1).long()
    v0 = vs.round().clamp_(0, H-1).long()
    mask_syn_sparse = torch.zeros((H, W), dtype=torch.float32, device=device)
    mask_syn_sparse[v0, u0] = 1.0

    return us, vs, idxs, mask_syn_sparse