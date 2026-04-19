import os, sys
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from run_nerf_helpers import *
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import pil_to_tensor, to_pil_image

to_tensor = transforms.ToTensor()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DEPTH_ANYTHING_ROOT = os.path.join(THIS_DIR, "Depth-Anything-V2", "metric_depth")
if DEPTH_ANYTHING_ROOT not in sys.path:
    sys.path.insert(0, DEPTH_ANYTHING_ROOT)

from depth_anything_v2.dpt import DepthAnythingV2


def neighbor_indices(t: int, n_frames: int, k: int, step: int = 1):
    """
    Return up to `k` neighbour indices around `t`, expanding outward by `step`.
    Always excludes `t` itself and skips indices outside [0, n_frames-1].

    Examples
    --------
    >>> neighbor_indices(0, n_frames=36, k=2, step=1)
    [1, 2]
    >>> neighbor_indices(10, n_frames=15, k=4, step=2)
    [8, 12, 6, 14]
    """
    neighbours = []
    offset = step
    while len(neighbours) < k:
        neg = t - offset
        if 0 <= neg < n_frames:
            neighbours.append(neg)
            if len(neighbours) == k:
                break
        pos = t + offset
        if 0 <= pos < n_frames:
            neighbours.append(pos)
            if len(neighbours) == k:
                break
        offset += step
    return neighbours


def weights_map_to_mask(weights_map_dd, mode="quantile", q=0.80, thr=0.5):
    """
    weights_map_dd: torch.Tensor [H,W] or [H,W,1]
    Returns: np.uint8 mask [H,W] (0/255)
    """
    if weights_map_dd.ndim == 3 and weights_map_dd.shape[-1] == 1:
        w = weights_map_dd[..., 0]
    else:
        w = weights_map_dd
    w = w.detach().float().cpu().numpy()
    w01 = (w - w.min()) / (w.max() - w.min() + 1e-8)

    if mode == "quantile":
        cut = np.quantile(w01, q)
        m = (w01 >= cut).astype(np.uint8) * 255
    else:  # "threshold"
        m = (w01 >= thr).astype(np.uint8) * 255

    return m


class DifixUpdater:
    def __init__(self, H, W, focal, render_fn, render_kwargs_test,
                 base_images, base_poses, k, step, save_root, data_dir, use_depth=False,
                 mask_warp=False, motion_masks=None, weights_map_ray=False, use_mv_mask=False,
                 inter_pose=False, sam2_predictor=None):
        self.H, self.W, self.focal = H, W, focal
        self.render = render_fn
        self.kw_test = render_kwargs_test
        self.base_imgs = base_images
        self.base_poses = base_poses
        self.k = k          # number of synthetic images per frame
        self.step = step
        self.save_root = save_root
        self.data_dir = data_dir
        self.synthetic_imgs = None
        self.synthetic_poses = None

        # Depth
        self.use_depth = use_depth
        self.synthetic_depths = None

        # Mask warp
        self.mask_warp = mask_warp
        self.base_motion_masks = motion_masks

        # Weights map ray
        self.weights_map_ray = weights_map_ray
        self.motion_coords = None
        self.use_mv_mask = use_mv_mask

        # SAM2 automatic masking
        self.sam2_predictor = sam2_predictor

        # Synthetic pose option
        self.inter_pose = inter_pose

    def _load_gt_mv_mask(self, t, n, dilate_ks=3, dilate_iter=2):
        """
        Load GT multi-view motion mask and apply dilation.
        Returns (mask_uint8, coords_int64) or (None, None) if file not found.
        """
        gt_path = os.path.join(self.data_dir, "mv_masks", f"{t:05d}", f"{n%9+1:05d}.png")
        mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return None, None

        mask = (mask > 0).astype(np.uint8) * 255
        kernel = np.ones((dilate_ks, dilate_ks), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=dilate_iter)
        mask_resized = cv2.resize(mask, (self.W, self.H), interpolation=cv2.INTER_NEAREST)
        mask_resized = (mask_resized > 0).astype(np.uint8) * 255

        ys, xs = np.where(mask_resized > 0)
        if ys.size == 0:
            coords = np.empty((0, 2), dtype=np.int64)
        else:
            coords = np.stack((ys, xs), axis=-1).astype(np.int64)

        return mask_resized, coords

    def _check_renders_done(self, step: int) -> bool:
        """Return True if all novel view images are already rendered for this step."""
        N = self.base_imgs.shape[0]
        for t in range(N):
            for n in neighbor_indices(t, N, self.k, self.step):
                img_path = os.path.join(
                    self.save_root, f'step_{step:06d}', f't_{t:03d}', f'n_{n:03d}.png')
                if not os.path.exists(img_path):
                    return False
        return True

    def _check_sam2_masking_done(self, step: int) -> bool:
        """Return True if all rendered novel views already have SAM2 coords on disk."""
        N = self.base_imgs.shape[0]
        for t in range(N):
            for n in neighbor_indices(t, N, self.k, self.step):
                outdir = os.path.join(self.save_root, f'step_{step:06d}', f't_{t:03d}')
                img_path    = os.path.join(outdir, f'n_{n:03d}.png')
                coords_path = os.path.join(outdir, f'n_{n:03d}_weights_coords.npy')
                if os.path.exists(img_path) and not os.path.exists(coords_path):
                    return False
        return True

    def _run_sam2_masking(self, step: int) -> None:
        """Run SAM2 on all rendered novel views to generate dynamic masks automatically.

        Images are grouped by neighbor index n (view-major order), giving temporal
        continuity within each group for reliable SAM2 propagation.

        Prompt strategy: bounding box extracted from base_motion_masks[n_view].
          - Novel view (t, n) is rendered from CAMERA n's pose.
          - motion_masks[n] is the real training frame captured from the same camera n.
          - Therefore motion_masks[n] gives the most accurate 2D bounding box of the
            dynamic object as seen from that camera — regardless of which time t is
            being rendered.
          - Using motion_masks[t_start] (different camera) would give a box in the
            wrong image region, causing SAM2 to lock onto background.

        Saves results to:
          {save_root}/step_{step:06d}/t_{t:03d}/n_{n:03d}_weights_map_mask.png
          {save_root}/step_{step:06d}/t_{t:03d}/n_{n:03d}_weights_coords.npy
        """
        if self.sam2_predictor is None:
            return
        if self.base_motion_masks is None:
            print('[SAM2] No motion_masks available — skipping SAM2 masking.')
            return

        import tempfile, shutil
        from PIL import Image as PILImage

        N = self.base_imgs.shape[0]

        # Collect (t, paths) grouped by view index n — skip already-masked pairs
        groups: dict = {}  # n -> [(t, img_path, mask_path, coords_path), ...]
        for t in range(N):
            for n in neighbor_indices(t, N, self.k, self.step):
                outdir      = os.path.join(self.save_root, f'step_{step:06d}', f't_{t:03d}')
                img_path    = os.path.join(outdir, f'n_{n:03d}.png')
                mask_path   = os.path.join(outdir, f'n_{n:03d}_weights_map_mask.png')
                coords_path = os.path.join(outdir, f'n_{n:03d}_weights_coords.npy')
                if not os.path.exists(img_path):
                    continue  # not rendered yet
                if os.path.exists(coords_path):
                    continue  # already processed
                groups.setdefault(n, []).append((t, img_path, mask_path, coords_path))

        if not groups:
            print('[SAM2] All masks already present — skipping.')
            return

        n_total = sum(len(v) for v in groups.values())
        print(f'[SAM2] Masking {n_total} novel views across {len(groups)} view group(s)...')

        kernel = np.ones((3, 3), np.uint8)

        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            for n_view, pairs in sorted(groups.items()):
                pairs_sorted = sorted(pairs, key=lambda x: x[0])  # sort by t

                # Bounding box from motion_masks[n_view]:
                #   Novel view uses camera n's pose → motion_masks[n] is the real frame
                #   from that same camera, giving the correct 2D object region.
                mm = self.base_motion_masks[n_view]  # [H, W]
                mm_np = (mm.detach().cpu().float().numpy()
                         if torch.is_tensor(mm) else np.asarray(mm, dtype=float))
                ys, xs = np.where(mm_np > 0.5)
                if ys.size == 0:
                    print(f'[SAM2]   n={n_view}: empty motion_mask[{n_view}] — skip.')
                    continue

                # Bounding box [x_min, y_min, x_max, y_max] with 10% margin
                H_img, W_img = mm_np.shape
                margin_y = max(1, int((ys.max() - ys.min()) * 0.10))
                margin_x = max(1, int((xs.max() - xs.min()) * 0.10))
                box = np.array([
                    max(0,     xs.min() - margin_x),
                    max(0,     ys.min() - margin_y),
                    min(W_img, xs.max() + margin_x),
                    min(H_img, ys.max() + margin_y),
                ], dtype=np.float32)

                # Build temp JPEG dir (SAM2 load_video_frames requires JPEG)
                tmp_dir = tempfile.mkdtemp(prefix='sam2_difix_')
                try:
                    for frame_idx, (_, img_path, _, _) in enumerate(pairs_sorted):
                        PILImage.open(img_path).convert('RGB').save(
                            os.path.join(tmp_dir, f'{frame_idx:05d}.jpg'), quality=95)

                    inf_state = self.sam2_predictor.init_state(video_path=tmp_dir)
                    self.sam2_predictor.reset_state(inf_state)
                    self.sam2_predictor.add_new_points_or_box(
                        inference_state=inf_state,
                        frame_idx=0,
                        obj_id=1,
                        box=box,
                    )

                    frame_masks: dict = {}
                    for out_idx, _, out_logits in self.sam2_predictor.propagate_in_video(inf_state):
                        frame_masks[out_idx] = (out_logits[0, 0] > 0).cpu().numpy().astype(np.uint8) * 255

                    for frame_idx, (_, _, mask_path, coords_path) in enumerate(pairs_sorted):
                        raw = frame_masks.get(frame_idx)
                        if raw is None:
                            continue
                        mask = cv2.dilate(raw, kernel, iterations=2)
                        cv2.imwrite(mask_path, mask)
                        ys2, xs2 = np.where(mask > 0)
                        coords = (np.stack((ys2, xs2), axis=-1).astype(np.int64)
                                  if ys2.size > 0 else np.empty((0, 2), np.int64))
                        np.save(coords_path, coords)

                    print(f'[SAM2]   n={n_view}: box={box.astype(int).tolist()}, '
                          f'{len(pairs_sorted)} frames done.')
                finally:
                    shutil.rmtree(tmp_dir, ignore_errors=True)

        print('[SAM2] Masking complete.')

    @torch.no_grad()
    def update(self, step, pipe, ref_imgs, cam_param, gt=True, gt_ratio=1.):
        """Generate Difix-enhanced synthetic views for all training frames (difix_ver2)."""
        N = self.base_imgs.shape[0]
        syn_imgs, syn_poses = [], []
        if self.weights_map_ray:
            self.motion_coords = {t: [] for t in range(N)}

        for t in range(N):
            neigh = neighbor_indices(t, N, self.k, self.step)
            for n in neigh:
                outdir = f"{self.save_root}/step_{step:06d}/t_{t:03d}"
                img_path = f"{outdir}/n_{n:03d}.png"
                mask_path = os.path.join(outdir, f"n_{n:03d}_weights_map_mask.png")
                coords_path = os.path.join(outdir, f"n_{n:03d}_weights_coords.npy")

                if self.inter_pose:
                    pose = (self.base_poses[n] + self.base_poses[t]) / 2
                pose = self.base_poses[n].to(device)

                needs_ret = self.mask_warp or (
                    self.weights_map_ray
                    and not (os.path.exists(coords_path) or os.path.exists(mask_path))
                )

                ret = None
                rgb = None

                print(f'Time: {t}, n:{n}')

                if needs_ret:
                    img_idx_embed = t / float(N) * 2. - 1.0
                    ret = self.render(img_idx_embed, 0, False, float(N),
                                      self.H, self.W, self.focal,
                                      chunk=1024*16, c2w=pose,
                                      **self.kw_test)
                    rgb = cam_param.RAD2LDR_img(ret['rgb_map_ref'], t).permute(2, 0, 1)

                if os.path.exists(img_path):
                    # Cache hit: load previously generated image
                    pil_img = Image.open(img_path).convert("RGB")
                    enh_tensor = (pil_to_tensor(pil_img).float() / 255.0).permute(1, 2, 0)

                    if self.weights_map_ray:
                        if os.path.exists(coords_path):
                            coords = np.load(coords_path)
                            if coords.ndim != 2 or coords.shape[-1] != 2:
                                coords = np.empty((0, 2), dtype=np.int64)
                            else:
                                coords = coords.astype(np.int64, copy=False)
                            self.motion_coords[t].append(coords)
                        elif os.path.exists(mask_path):
                            mask_png = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                            if mask_png is None:
                                self.motion_coords[t].append(np.empty((0, 2), dtype=np.int64))
                            else:
                                ys, xs = np.where(mask_png > 0)
                                coords = (np.stack((ys, xs), axis=-1).astype(np.int64)
                                          if ys.size > 0 else np.empty((0, 2), np.int64))
                                self.motion_coords[t].append(coords)
                                np.save(coords_path, coords)
                        else:
                            self.motion_coords[t].append(np.empty((0, 2), dtype=np.int64))

                else:
                    if rgb is None:
                        print("Difix: no RGB available, skipping")
                        return

                    if gt:
                        gt_img_path = os.path.join(self.data_dir, "images", f"{t:05d}.png")
                        gt_img = Image.open(gt_img_path).convert("RGB")
                        gt_tensor = (pil_to_tensor(gt_img).float() / 255.).to(rgb.device)

                        if gt_ratio != 1.:
                            new_H = int(gt_tensor.shape[1] * gt_ratio)
                            new_W = int(gt_tensor.shape[2] * gt_ratio)
                            gt_tensor = F.interpolate(
                                gt_tensor.unsqueeze(0),
                                size=(new_H, new_W),
                                mode="bilinear", align_corners=False
                            ).squeeze(0).clamp(0, 1)

                        rgb_resized = F.interpolate(
                            rgb.unsqueeze(0),
                            size=(gt_tensor.shape[1], gt_tensor.shape[2]),
                            mode="bilinear", align_corners=False
                        ).squeeze(0).clamp(0, 1)

                        enh = pipe(
                            "remove degradation",
                            image=rgb_resized,
                            ref_image=gt_tensor,
                            num_inference_steps=1,
                            timesteps=[199],
                            guidance_scale=0.0,
                        ).images[0]
                    else:
                        enh = pipe(
                            "remove degradation",
                            image=rgb,
                            ref_image=ref_imgs[t].permute(2, 0, 1),
                            num_inference_steps=1,
                            timesteps=[199],
                            guidance_scale=0.0,
                        ).images[0]

                    os.makedirs(outdir, exist_ok=True)
                    enh_resized = enh.resize((rgb.shape[2], rgb.shape[1]), Image.BILINEAR)
                    enh_resized.save(img_path)

                    rgb_pil = to_pil_image(rgb.cpu().clamp(0, 1))
                    rgb_pil.save(f"{outdir}/n_{n:03d}_o.png")

                    enh_tensor = pil_to_tensor(enh_resized).to(rgb.device).permute(1, 2, 0) / 255.

                    if self.mask_warp:
                        # Get ref-view depth for mask reprojection
                        if ret is None or pose is not self.base_poses[t]:
                            pose_ref = self.base_poses[t].to(device)
                            img_idx_embed = t / float(N) * 2. - 1.0
                            ret_ref = self.render(img_idx_embed, 0, False, float(N),
                                                  self.H, self.W, self.focal,
                                                  chunk=1024*16, c2w=pose_ref,
                                                  **self.kw_test)
                            depth_ref = ret_ref['depth_map_ref'].detach()
                        else:
                            depth_ref = ret['depth_map_ref'].detach()

                        us, vs, idxs, mask_syn = project_mask_points_v2(
                            self.base_motion_masks[t],
                            depth_ref,
                            self.base_poses[t, :3, :4],
                            self.base_poses[n, :3, :4],
                            self.H, self.W, self.focal,
                            depth_mode='ndc_t',
                            near=1.0,
                            return_sparse_mask=True,
                        )
                        mask_np = (mask_syn.cpu().numpy() * 255).astype(np.uint8)
                        kernel = np.ones((3, 3), np.uint8)
                        dilated = cv2.dilate(mask_np, kernel, iterations=2)
                        cv2.imwrite(f"{outdir}/n_{n:03d}_mask.png", dilated)

                    elif self.weights_map_ray:
                        wrote_coords = False

                        if self.use_mv_mask:
                            gt_mask, gt_coords = self._load_gt_mv_mask(t, n)
                            if gt_mask is not None:
                                cv2.imwrite(mask_path, gt_mask)
                                np.save(coords_path, gt_coords)
                                self.motion_coords[t].append(gt_coords)
                                wrote_coords = True
                                print(f"[MV-GT] t={t}, n={n} -> {gt_coords.shape[0]} coords")

                        if not wrote_coords and ret is not None and 'weights_map_dd' in ret:
                            mask_png = weights_map_to_mask(
                                ret['weights_map_dd'], mode='threshold', q=0.80, thr=0.99)
                            cv2.imwrite(mask_path, mask_png)
                            if self.sam2_predictor is None:
                                # No SAM2: use (noisy) weights_map coords directly
                                ys, xs = np.where(mask_png > 0)
                                coords = (np.stack((ys, xs), axis=-1).astype(np.int64)
                                          if ys.size > 0 else np.empty((0, 2), np.int64))
                                self.motion_coords[t].append(coords)
                                np.save(coords_path, coords)
                            else:
                                # SAM2 will write coords_path; use empty for this pass
                                self.motion_coords[t].append(np.empty((0, 2), np.int64))
                            wrote_coords = True

                        if not wrote_coords and os.path.exists(coords_path) \
                                and len(self.motion_coords[t]) < len(neigh):
                            coords = np.load(coords_path)
                            if coords.ndim != 2 or coords.shape[-1] != 2:
                                coords = np.empty((0, 2), dtype=np.int64)
                            else:
                                coords = coords.astype(np.int64, copy=False)
                            self.motion_coords[t].append(coords)

                # Resize to (H, W) if needed
                h, w, _ = enh_tensor.shape
                if (h, w) != (self.H, self.W):
                    if h == self.H and w == self.W - 1:
                        enh_tensor = F.pad(enh_tensor.permute(2, 0, 1),
                                           (0, 1, 0, 0), mode='replicate').permute(1, 2, 0)
                    else:
                        enh_tensor = F.interpolate(
                            enh_tensor.permute(2, 0, 1).unsqueeze(0),
                            size=(self.H, self.W),
                            mode='bilinear', align_corners=False
                        )[0].permute(1, 2, 0)

                syn_imgs.append(enh_tensor.cpu())
                syn_poses.append(pose.cpu())

        self.synthetic_imgs = torch.stack(syn_imgs)
        self.synthetic_poses = torch.stack(syn_poses)

        if self.use_depth:
            viz_root = f"{self.save_root}/step_{step:06d}"
            self.compute_synthetic_depths(
                encoder='vitl',
                ckpt_path='./Depth-Anything-V2/metric_depth/checkpoints/depth_anything_v2_metric_hypersim_vitl.pth',
                input_size=518,
                max_depth=40.0,
                save_numpy=True,
                save_viz=True,
                viz_grayscale=False,
                pred_only=True,
                viz_root=viz_root,
            )

    def init_depth_model(self, encoder: str, ckpt_path: str, max_depth: float, device: str):
        """Load Depth-Anything V2 model."""
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64,  'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]},
        }
        if encoder not in model_configs:
            raise ValueError(f"encoder must be one of {list(model_configs.keys())}")
        depth_anything = DepthAnythingV2(**{**model_configs[encoder], 'max_depth': max_depth})
        state = torch.load(ckpt_path, map_location='cpu')
        depth_anything.load_state_dict(state)
        return depth_anything.to(device).eval()

    @torch.no_grad()
    def compute_synthetic_depths(
        self,
        encoder: str = 'vitl',
        ckpt_path: str = './Depth-Anything-V2/metric_depth/checkpoints/depth_anything_v2_metric_hypersim_vitl.pth',
        input_size: int = 518,
        max_depth: float = 40.0,
        save_numpy: bool = False,
        save_viz: bool = False,
        viz_grayscale: bool = False,
        pred_only: bool = True,
        viz_root: str = None,
    ):
        """
        Run Depth-Anything V2 on self.synthetic_imgs and store results in
        self.synthetic_depths (FloatTensor [M, H, W]).
        """
        if self.synthetic_imgs is None:
            raise RuntimeError("Call update() before compute_synthetic_depths().")

        DEVICE = ('cuda' if torch.cuda.is_available() else
                  'mps' if torch.backends.mps.is_available() else 'cpu')
        depth_anything = self.init_depth_model(encoder, ckpt_path, max_depth, DEVICE)

        syn = self.synthetic_imgs
        syn_list = [syn[i] for i in range(syn.shape[0])] if not isinstance(syn, list) else syn

        depths = []
        if (save_numpy or save_viz) and viz_root is not None:
            import matplotlib
            npy_dir = os.path.join(viz_root, 'depth-anything')
            viz_dir = os.path.join(viz_root, 'depth-anything-viz')
            if save_numpy: os.makedirs(npy_dir, exist_ok=True)
            if save_viz:   os.makedirs(viz_dir, exist_ok=True)
            cmap = matplotlib.colormaps.get_cmap('Spectral')

        for idx, img in enumerate(syn_list):
            img_np = (img.detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8) \
                     if torch.is_tensor(img) else (np.asarray(img) * 255.0).clip(0, 255).astype(np.uint8)
            bgr = img_np[:, :, ::-1]
            depth = depth_anything.infer_image(bgr, input_size)
            depths.append(torch.from_numpy(depth).float())

            if (save_numpy or save_viz) and viz_root is not None:
                base = f"syn_{idx:05d}"
                if save_numpy:
                    np.save(os.path.join(npy_dir, base + ".npy"), depth)
                if save_viz:
                    dmin, dmax = depth.min(), depth.max()
                    dvis = ((depth - dmin) / (dmax - dmin) * 255.0
                            if dmax > dmin else np.zeros_like(depth)).astype(np.uint8)
                    if viz_grayscale:
                        dvis_rgb = np.repeat(dvis[..., None], 3, axis=-1)
                    else:
                        dvis_rgb = (cmap(dvis)[:, :, :3] * 255).astype(np.uint8)[:, :, ::-1]
                    if pred_only:
                        cv2.imwrite(os.path.join(viz_dir, base + ".png"), dvis_rgb)
                    else:
                        split = np.ones((bgr.shape[0], 50, 3), dtype=np.uint8) * 255
                        cv2.imwrite(os.path.join(viz_dir, base + ".png"),
                                    cv2.hconcat([bgr, split, dvis_rgb]))

        self.synthetic_depths = torch.stack(depths, dim=0)
