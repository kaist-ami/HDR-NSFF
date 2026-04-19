import os, sys
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import numpy as np
import time
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import cv2
from kornia import create_meshgrid
from render_utils import *
from run_nerf_helpers import *
from load_llff import *
from load_blender import *
import cam_param
import wandb
from tqdm import tqdm
import imageio.v3 as iio
from percept import *
from difix_updater import *
from config import config_parser

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(1)

img2mse = lambda x, y: torch.mean((x - y) ** 2)
img2mae = lambda x, y: torch.mean(torch.abs(x - y))


def train():

    parser = config_parser()
    args = parser.parse_args()

    # ── Data Loading ───────────────────────────────────────────────────────────
    if args.dataset_type == 'llff':
        target_idx = args.target_idx
        images, depths, masks, poses, bds, \
        render_poses, ref_c2w, motion_coords = load_llff_data(
            args.datadir,
            args.start_frame, args.end_frame, args.step_frame,
            args.factor,
            target_idx=target_idx,
            recenter=True, bd_factor=.9,
            spherify=args.spherify,
            final_height=args.final_height,
            disp_model=args.disp_model,
        )
        print('Loaded disp model:', args.disp_model)

        hwf = poses[0, :3, -1]
        poses = poses[:, :3, :4]
        print('Loaded llff', images.shape, render_poses.shape, hwf, args.datadir)

        i_test = []
        i_val = []
        i_train = np.array([i for i in np.arange(int(images.shape[0]))
                            if i not in i_test and i not in i_val])

        if args.no_ndc:
            near = np.percentile(bds[:, 0], 5) * 0.8
            far  = np.percentile(bds[:, 1], 95) * 1.1
        else:
            near, far = 0., 1.
        print('NEAR FAR', near, far)

    elif args.dataset_type == 'blender':
        target_idx = args.target_idx
        images, poses, render_poses, hwf, i_split, \
        depths, masks, ref_c2w, motion_coords, near, far = load_blender_data(
            args.datadir,
            args.start_frame, args.end_frame,
            args.half_res, args.testskip,
            factor=args.factor,
            target_idx=target_idx,
            final_height=args.final_height,
            disp_model=args.disp_model,
        )
        i_val = []
        i_test = []
        i_train = np.array([i for i in np.arange(int(images.shape[0]))
                            if i not in i_test and i not in i_val])
        images = images[..., :3]

    else:
        print('Unknown dataset type:', args.dataset_type)
        sys.exit()

    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    # ── Experiment Directory Setup ─────────────────────────────────────────────
    basedir = args.basedir
    args.expname = args.expname + '_F%02d-%02d' % (args.start_frame, args.end_frame)
    expname = args.expname

    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    with open(os.path.join(basedir, expname, 'args.txt'), 'w') as f:
        for arg in sorted(vars(args)):
            f.write('{} = {}\n'.format(arg, getattr(args, arg)))
    if args.config is not None:
        with open(os.path.join(basedir, expname, 'config.txt'), 'w') as f:
            f.write(open(args.config, 'r').read())

    # ── Model Creation ─────────────────────────────────────────────────────────
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)
    global_step = start

    bds_dict = {'near': near, 'far': far}
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    # ── Camera Parameters (HDR tone mapping / CRF / white balance) ────────────
    if args.use_tone_mapping:
        N, H_img, W_img, C = images.shape
        Cam_param = cam_param.CamParam(
            N, H_img, W_img, device=device, gts=images,
            initialize=args.use_initialize,
            tone_mapping=args.tone_mapping,
            share_crf=args.share_crf,
            share_wb=args.share_wb,
            ref_idx=args.ref_idx,
            log_scale=args.log_scale,
        )
        optim_cam_wb, optim_cam_crf = Cam_param.optimizer(l_rate=args.lrate)
        camparam_frozen = False
        crf_stop_step = args.crf_stop_step

        # Load CamParam checkpoint if available
        if args.ft_path is not None and args.ft_path != 'None':
            ckpts = [args.ft_path]
        else:
            ckpts = [os.path.join(basedir, expname, f)
                     for f in sorted(os.listdir(os.path.join(basedir, expname))) if 'tar' in f]

        if len(ckpts) > 0 and not args.no_reload:
            ckpt_path = ckpts[-1]
            print('Reloading CamParam from', ckpt_path)
            ckpt = torch.load(ckpt_path)
            Cam_param.load_ckpt(ckpt)
            optim_cam_wb.load_state_dict(ckpt['optim_cam_wb_state_dict'])
            if ckpt['optim_cam_crf_state_dict'] is not None:
                if optim_cam_crf is None:
                    optim_cam_crf = torch.optim.Adam(
                        Cam_param.crf_params, lr=args.lrate, betas=(0.9, 0.999))
                optim_cam_crf.load_state_dict(ckpt['optim_cam_crf_state_dict'])

        # Visualize CRF curves and exit (no training)
        if args.viz_crf:
            viz_crf_dir = os.path.join(basedir, expname, 'crf_viz')
            os.makedirs(viz_crf_dir, exist_ok=True)
            for i in range(int(images.shape[0])):
                Cam_param.visualize_crf(i, os.path.join(viz_crf_dir, f'crf_{i:05d}.png'))
            Cam_param.save_wb_plot(save_path=os.path.join(viz_crf_dir, 'wb.png'))
            sys.exit(0)
    else:
        Cam_param       = None
        optim_cam_crf   = None
        optim_cam_wb    = None
        camparam_frozen = False
        crf_stop_step   = 0

    # ── Inference-Only Render Modes ────────────────────────────────────────────
    # Each block renders output and exits without training.

    if args.render_bt:
        # Space-time interpolation: sweep poses at a fixed timestamp
        render_poses = torch.Tensor(render_poses).to(device)
        num_img = float(poses.shape[0])
        img_idx_embed = target_idx / float(num_img) * 2. - 1.0
        testsavedir = os.path.join(
            basedir, expname,
            'render-spiral-frame-%03d' % target_idx
            + '_{}_{:06d}'.format('test' if args.render_test else 'path', start))
        with torch.no_grad():
            render_bullet_time(render_poses, img_idx_embed, num_img, hwf,
                               args.chunk, render_kwargs_test,
                               args.render_mode, args.render_tm,
                               gt_imgs=images, savedir=testsavedir,
                               camparam=Cam_param, render_factor=args.render_factor)
        return

    if args.render_lockcam_slowmo:
        # Time interpolation at a single locked camera pose
        num_img  = float(poses.shape[0])
        ref_c2w  = torch.Tensor(ref_c2w).to(device)
        testsavedir = os.path.join(basedir, expname, 'render-lockcam-slowmo')
        with torch.no_grad():
            render_lockcam_slowmo(ref_c2w, num_img, hwf,
                                  args.chunk, render_kwargs_test,
                                  gt_imgs=images, savedir=testsavedir,
                                  render_factor=args.render_factor,
                                  target_idx=target_idx,
                                  camparam=Cam_param, render_tm=args.render_tm)
        return

    if args.render_lockcam_slowmo_full:
        # Full-sequence time interpolation at a locked camera pose
        num_img  = float(poses.shape[0])
        ref_c2w  = torch.Tensor(ref_c2w).to(device)
        testsavedir = os.path.join(basedir, expname,
                                   f'render-lockcam-slowmo-full-{target_idx:02d}')
        with torch.no_grad():
            render_lockcam_slowmo_full(ref_c2w, num_img, hwf,
                                       args.chunk, render_kwargs_test,
                                       gt_imgs=images, savedir=testsavedir,
                                       render_factor=args.render_factor,
                                       target_idx=target_idx,
                                       camparam=Cam_param, render_tm=args.render_tm)
        return

    if args.render_slowmo_bt:
        # Slow-motion bullet-time rendering
        bt_poses     = create_bt_poses(hwf) * 10
        custom_times = args.custom_times if args.custom_times else None
        testsavedir  = os.path.join(
            basedir, expname,
            'render-slowmo_bt_{}_{:06d}_{}'.format(
                'test' if args.render_test else 'path', start, str(args.render_mode)))
        images = torch.Tensor(images)
        with torch.no_grad():
            render_slowmo_bt(depths, poses, bt_poses,
                             hwf, args.chunk, render_kwargs_test,
                             args.render_mode, args.render_tm,
                             gt_imgs=images, savedir=testsavedir,
                             camparam=Cam_param,
                             render_factor=args.render_factor,
                             target_idx=args.target_idx,
                             custom_times=custom_times,
                             start=start)
        return

    if args.render_train:
        # Render all training views and save HDR EXR, tone-mapped HDR PNG, and per-exposure LDR
        num_img          = poses.shape[0]
        testsavedir      = os.path.join(basedir, expname, 'render-train_{:06d}'.format(start))
        save_hdr_tm_dir  = os.path.join(testsavedir, 'hdr_tm')
        save_hdr_exr_dir = os.path.join(testsavedir, 'hdr_exr')
        save_ldr_dir     = os.path.join(testsavedir, 'ldr')
        os.makedirs(save_hdr_tm_dir,  exist_ok=True)
        os.makedirs(save_hdr_exr_dir, exist_ok=True)
        os.makedirs(save_ldr_dir,     exist_ok=True)
        mu = torch.tensor(50).to(device)
        with torch.no_grad():
            for img_i in range(num_img):
                print(f'Rendering train frame {img_i}/{num_img}')
                img_idx_embed = img_i / num_img * 2. - 1.0
                pose = torch.Tensor(poses[img_i, :3, :4]).to(device)
                ret  = render(img_idx_embed, 0, False, num_img,
                              H, W, focal, chunk=1024 * 16, c2w=pose,
                              **render_kwargs_test)
                rgb_hdr = ret['rgb_map_ref']
                # Mu-law tone mapping for HDR preview
                hdr_tm = torch.log(1 + mu * rgb_hdr) / torch.log(1 + mu)
                imageio.imwrite(
                    os.path.join(save_hdr_tm_dir, f'{img_i:05d}.png'),
                    (torch.clamp(hdr_tm, 0., 1.) * 255).cpu().numpy().astype(np.uint8))
                # HDR radiance as EXR (OpenCV expects BGR)
                cv2.imwrite(
                    os.path.join(save_hdr_exr_dir, f'{img_i:05d}.exr'),
                    rgb_hdr.cpu().numpy().astype(np.float32)[..., ::-1])
                # Per-exposure LDR images via learned CRF
                if Cam_param is not None:
                    for exp_idx in range(3):
                        ldr = to8b(Cam_param.RAD2LDR_img(rgb_hdr, exp_idx).cpu().numpy())
                        imageio.imwrite(
                            os.path.join(save_ldr_dir, f'{img_i:05d}_{exp_idx}.png'), ldr)
        return

    if args.render_dy:
        # Render dynamic decomposition (ref / post / prev) for each training frame
        num_img      = poses.shape[0]
        testsavedir  = os.path.join(basedir, expname, 'render-train_dy_{:06d}'.format(start))
        save_img_dir = os.path.join(testsavedir, 'images')
        os.makedirs(save_img_dir, exist_ok=True)
        render_kwargs_test['inference'] = False
        with torch.no_grad():
            for img_i in tqdm(range(num_img), desc='Rendering dynamic frames'):
                img_idx_embed = img_i / num_img * 2. - 1.0
                pose = torch.from_numpy(poses[img_i, :3, :4]).to(device)
                ret  = render(img_idx_embed, 0, False, num_img,
                              H, W, focal, chunk=1024 * 16, c2w=pose,
                              **render_kwargs_test)
                for suffix, key in [('ref',  'rgb_map_ref_dy'),
                                     ('post', 'rgb_map_post_dy'),
                                     ('prev', 'rgb_map_prev_dy')]:
                    rgb = (torch.clamp(
                        Cam_param.RAD2LDR_img(ret[key], img_i), 0., 1.) * 255
                    ).cpu().numpy().astype(np.uint8)
                    imageio.imwrite(os.path.join(save_img_dir, f'{img_i:05d}_{suffix}.png'), rgb)
        return

    if args.render_interpolate:
        # Soft-splatting-based temporal interpolation between adjacent training frames.
        # Blends forward/backward renders at ratio=0.5 to produce mid-frame images.
        poses_t  = torch.Tensor(poses).to(device)
        num_img  = poses_t.shape[0]
        render_dir = os.path.join(basedir, expname, 'render_inter_mid_exp')
        os.makedirs(os.path.join(render_dir, 'hdr_tm'),  exist_ok=True)
        os.makedirs(os.path.join(render_dir, 'ldr'),     exist_ok=True)
        os.makedirs(os.path.join(render_dir, 'hdr_exr'), exist_ok=True)
        with torch.no_grad():
            for img_i in i_train[:-1]:
                ratio = 0.5
                img_idx_embed_1 = img_i / num_img * 2. - 1.0
                img_idx_embed_2 = (img_i + 1) / num_img * 2. - 1.0

                pose_1      = poses_t[img_i]
                pose_2      = poses_t[img_i + 1]
                render_pose = (pose_1 + pose_2) / 2
                R_w2t = render_pose[:3, :3].transpose(0, 1)
                t_w2t = -torch.matmul(R_w2t, render_pose[:3, 3:4])

                ret1 = render_sm(img_idx_embed_1, 0, False, num_img,
                                 H, W, focal, chunk=1024 * 16,
                                 c2w=render_pose, **render_kwargs_test)
                ret2 = render_sm(img_idx_embed_2, 0, False, num_img,
                                 H, W, focal, chunk=1024 * 16,
                                 c2w=render_pose, **render_kwargs_test)

                T_i       = torch.ones((1, H, W))
                final_rgb = torch.zeros((3, H, W))
                num_sample = ret1['raw_rgb'].shape[2]

                for j in range(num_sample):
                    sa_dy1, sr_dy1, sa_rig1, sr_rig1 = splat_rgb_img(
                        ret1, ratio,       R_w2t, t_w2t, j, H, W, focal, True)
                    sa_dy2, sr_dy2, sa_rig2, sr_rig2 = splat_rgb_img(
                        ret2, 1. - ratio,  R_w2t, t_w2t, j, H, W, focal, False)

                    final_rgb += T_i * (sa_dy1 * sr_dy1 + sa_rig1 * sr_rig1) * (1.0 - ratio)
                    final_rgb += T_i * (sa_dy2 * sr_dy2 + sa_rig2 * sr_rig2) * ratio

                    a1 = (1. - (1. - sa_dy1) * (1. - sa_rig1)) * (1. - ratio)
                    a2 = (1. - (1. - sa_dy2) * (1. - sa_rig2)) * ratio
                    T_i = T_i * (1.0 - (a1 + a2) + 1e-10)

                if Cam_param is not None:
                    # Snap to nearest reference exposure index
                    tone_map = img_i - (img_i % 3)
                    rgb8_ldr = to8b(Cam_param.RAD2LDR_img(
                        final_rgb.permute(1, 2, 0), tone_map).cpu().numpy())
                    mu = torch.tensor(50, dtype=final_rgb.dtype, device=final_rgb.device)
                    final_rgb_tm = torch.log(1 + mu * final_rgb) / torch.log(1 + mu)
                    imageio.imwrite(os.path.join(render_dir, 'hdr_tm', f'{img_i:05d}.png'),
                                    to8b(final_rgb_tm.permute(1, 2, 0).cpu().numpy()))
                    imageio.imwrite(os.path.join(render_dir, 'ldr', f'{img_i:05d}.png'), rgb8_ldr)
                    cv2.imwrite(os.path.join(render_dir, f'{img_i:05d}.exr'),
                                final_rgb.permute(1, 2, 0).cpu().numpy().astype(np.float32)[..., ::-1])
                else:
                    imageio.imwrite(os.path.join(render_dir, 'ldr', f'{img_i:05d}.png'),
                                    to8b(final_rgb.permute(1, 2, 0).cpu().numpy()))
        return

    if args.render_lockcam:
        # Fixed-camera rendering across all timesteps.
        # Saves: HDR EXR, mu-law tone-mapped HDR PNG, depth NPY, and (optionally) per-exposure LDR.
        num_img     = float(poses.shape[0])
        ref_c2w     = torch.Tensor(ref_c2w).to(device)
        testsavedir = os.path.join(basedir, expname,
                                   f'render-lockcam-{target_idx:03d}-{start:06d}')
        save_hdr_tm_dir  = os.path.join(testsavedir, 'hdr_tm')
        save_depth_dir   = os.path.join(testsavedir, 'depths')
        save_hdr_exr_dir = os.path.join(testsavedir, 'hdr_exr')
        os.makedirs(save_hdr_tm_dir,  exist_ok=True)
        os.makedirs(save_depth_dir,   exist_ok=True)
        os.makedirs(save_hdr_exr_dir, exist_ok=True)
        if Cam_param is not None and args.render_tm == 'wb':
            dir_ldr = [os.path.join(testsavedir, 'ldr', str(k)) for k in range(3)]
            for d in dir_ldr:
                os.makedirs(d, exist_ok=True)
        mu   = torch.tensor(50).to(device)
        pose = torch.Tensor(poses[target_idx, :3, :4]).to(device)
        with torch.no_grad():
            t = time.time()
            for img_i in range(int(num_img)):
                img_idx_embed = img_i / num_img * 2. - 1.0
                ret = render(img_idx_embed, 0, False, num_img,
                             H, W, focal, chunk=1024 * 16, c2w=pose,
                             **render_kwargs_test)
                np.save(os.path.join(save_depth_dir, f'{img_i:05d}_depth.npy'),
                        ret['depth_map_ref'].cpu().numpy())
                rgb_hdr = ret['rgb_map_ref']
                hdr_tm  = torch.log(1 + mu * rgb_hdr) / torch.log(1 + mu)
                cv2.imwrite(os.path.join(save_hdr_exr_dir, f'{img_i:05d}.exr'),
                            rgb_hdr.cpu().numpy().astype(np.float32)[..., ::-1])
                imageio.imwrite(os.path.join(save_hdr_tm_dir, f'{img_i:05d}.jpg'),
                                to8b(hdr_tm.cpu().numpy()))
                if Cam_param is not None and args.render_tm == 'wb':
                    for k, d in enumerate(dir_ldr):
                        ldr = to8b(Cam_param.RAD2LDR_img(rgb_hdr, k).cpu().numpy())
                        imageio.imwrite(os.path.join(d, f'{img_i:03d}_{k}.jpg'), ldr)
                print(f'lockcam {img_i:3d}/{int(num_img)} ({time.time()-t:.1f}s)')
                t = time.time()
        return

    # ── Training Setup ─────────────────────────────────────────────────────────
    N_rand = args.N_rand
    images = torch.Tensor(images)
    depths = torch.Tensor(depths)
    masks  = torch.Tensor(masks).to(device)
    poses  = torch.Tensor(poses).to(device)

    print('TRAIN views are', i_train)
    uv_grid = create_meshgrid(H, W, normalized_coordinates=False)[0].cuda()  # (H, W, 2)

    writer  = SummaryWriter(os.path.join(basedir, 'summaries', expname))
    num_img = float(images.shape[0])

    # Depth / flow loss weight decay schedule (capped to avoid over-decay on short sequences)
    decay_iteration = min(max(args.decay_iteration, args.end_frame - args.start_frame), 250)

    chain_bwd = 0
    l1_flag   = args.use_render_l1

    debug_mode = 1 if args.debug else 0
    if not debug_mode:
        wandb.init(project="HDR-NSFF", name=args.expname, resume="allow")

    if args.use_percept:
        perceptual_net = VGGPerceptual().to(device)
        perceptual_net.eval()
        patch_sz = args.patch_size
        N_rand   = patch_sz * patch_sz

    # Stage 2: disable geometry regularization losses (pure static refinement)
    if args.stage == 2:
        args.w_sf_reg       = 0
        args.w_cycle        = 0
        args.w_optical_flow = 0
        args.w_sm           = 0
        args.w_prob_reg     = 0
        args.w_depth        = 0
        args.w_entropy      = 0

    # ── Difix3D Setup ──────────────────────────────────────────────────────────
    # DifixPipeline is loaded lazily: initialized only when difix_start_step is
    # reached during training, so phase-1 (pre-difix) and phase-2 (difix) can
    # run in a single continuous training session without manual checkpoint copying.
    use_difix_flag = False
    if args.use_difix:
        difix_start_step = args.difix_start_step if args.difix_start_step is not None else 200000
        difix_save_root  = os.path.join(basedir, expname, 'difix_views')
        difix_pipe       = None  # loaded lazily at difix_start_step

        # ── SAM2 predictor for automatic novel-view masking ──────────────────
        # Loaded now (before training loop) so it is available when difix starts.
        # Falls back gracefully (sam2_predictor=None) if paths are missing.
        _nsff_exp_dir = os.path.dirname(os.path.abspath(__file__))
        _sam2_repo = (args.sam2_repo if args.sam2_repo
                      else os.path.join(_nsff_exp_dir, '..', 'nsff_scripts', 'sam2_repo'))
        _sam2_repo = os.path.normpath(_sam2_repo)
        _sam2_ckpt = (args.sam2_checkpoint if args.sam2_checkpoint
                      else os.path.join(_sam2_repo, 'checkpoints', 'sam2.1_hiera_large.pt'))
        sam2_predictor = None
        if os.path.isdir(_sam2_repo) and os.path.isfile(_sam2_ckpt):
            try:
                if _sam2_repo not in sys.path:
                    sys.path.insert(0, _sam2_repo)
                from sam2.build_sam import build_sam2_video_predictor
                sam2_predictor = build_sam2_video_predictor(
                    args.sam2_model_cfg, _sam2_ckpt, device=device)
                print(f'[Difix] SAM2 predictor loaded from {_sam2_ckpt}')
            except Exception as _e:
                print(f'[Difix] SAM2 load failed ({_e}); falling back to weights_map masks.')
        else:
            print(f'[Difix] SAM2 repo/checkpoint not found; falling back to weights_map masks.')

        difix_updater = DifixUpdater(
            H, W, focal, render, render_kwargs_test,
            images, poses,
            args.difix_num_views, args.difix_view_step,
            difix_save_root, args.datadir,
            use_depth=False, mask_warp=False,
            motion_masks=masks,
            weights_map_ray=True,
            use_mv_mask=(sam2_predictor is None),  # mv_mask only when no SAM2
            inter_pose=False,
            sam2_predictor=sam2_predictor,
        )

        images_orig = images
        poses_orig  = poses
        is_syn = 1

        if global_step >= difix_start_step:
            # Resuming a checkpoint taken after difix already started:
            # load the pipeline immediately and restore synthetic views.
            sys.path.append(os.path.join(os.path.dirname(__file__), "Difix3D"))
            from src.pipeline_difix import DifixPipeline
            difix_pipe = DifixPipeline.from_pretrained("nvidia/difix_ref", trust_remote_code=True)
            difix_pipe.to(device)

            difix_log_step = (global_step // args.difix_interval) * args.difix_interval
            next_refresh   = difix_log_step + args.difix_interval

            if os.path.exists(difix_save_root):
                if not difix_updater._check_renders_done(difix_log_step):
                    difix_updater.update(difix_log_step, difix_pipe, images_orig,
                                         Cam_param, True, args.difix_ref_scale)
                if not difix_updater._check_sam2_masking_done(difix_log_step):
                    difix_updater._run_sam2_masking(difix_log_step)
                difix_updater.update(difix_log_step, difix_pipe, images_orig,
                                     Cam_param, True, args.difix_ref_scale)
                images_syn = difix_updater.synthetic_imgs.to(images.device)
                poses_syn  = difix_updater.synthetic_poses.to(poses.device)
                images = torch.cat([images_orig, images_syn], 0)
                poses  = torch.cat([poses_orig,  poses_syn],  0)
        else:
            # Phase-1: difix not yet active; pipeline will be initialized lazily.
            next_refresh = difix_start_step

    # ── Training Loop ──────────────────────────────────────────────────────────
    for i in range(start, args.N_iters):
        freeze_cam_this_iter = False
        chain_bwd = 1 - chain_bwd
        time0 = time.time()


        if i % (decay_iteration * 1000) == 0:
            torch.cuda.empty_cache()

        # ── Difix Pipeline Lazy Init ───────────────────────────────────────────
        # First time we reach difix_start_step: load the pipeline and run the
        # initial novel-view synthesis + difix enhancement pass.
        if args.use_difix and difix_pipe is None and i >= difix_start_step:
            print(f'[Iter {i}] Initializing Difix pipeline...')
            sys.path.append(os.path.join(os.path.dirname(__file__), "Difix3D"))
            from src.pipeline_difix import DifixPipeline
            difix_pipe = DifixPipeline.from_pretrained("nvidia/difix_ref", trust_remote_code=True)
            difix_pipe.to(device)

            # Pass 1: render all novel views (skipped if already cached)
            if not difix_updater._check_renders_done(i):
                difix_updater.update(i, difix_pipe, images_orig,
                                     Cam_param, True, args.difix_ref_scale)
            # SAM2: generate masks (skipped if already done)
            difix_updater._run_sam2_masking(i)
            # Pass 2: fast cache hit, loads SAM2 coords into motion_coords
            difix_updater.update(i, difix_pipe, images_orig,
                                 Cam_param, True, args.difix_ref_scale)
            images_syn = difix_updater.synthetic_imgs.to(images.device)
            poses_syn  = difix_updater.synthetic_poses.to(poses.device)
            images = torch.cat([images_orig, images_syn], 0)
            poses  = torch.cat([poses_orig,  poses_syn],  0)
            next_refresh = i + args.difix_interval

        # ── Frame & Ray Sampling ───────────────────────────────────────────────
        img_i = np.random.choice(i_train)

        if args.use_difix and i > difix_start_step:
            use_difix_flag = True
            is_syn = 0 if (np.random.rand() < args.difix_gt_prob) else 1

            if is_syn != 0:
                which_syn   = np.random.choice(args.difix_num_views)
                img_i_difix = len(i_train) + img_i * args.difix_num_views + which_syn
                for p in render_kwargs_train['network_rigid'].parameters():
                    p.requires_grad = False
                render_kwargs_train['network_rigid'].eval()
            else:
                img_i_difix = img_i
                for p in render_kwargs_train['network_rigid'].parameters():
                    p.requires_grad = True
                render_kwargs_train['network_rigid'].train()

            target   = images[img_i_difix].cuda()
            pose     = poses[img_i_difix, :3, :4]
            depth_gt = depths[img_i].cuda()
        else:
            target   = images[img_i].cuda()
            pose     = poses[img_i, :3, :4]
            depth_gt = depths[img_i].cuda()

        hard_coords = torch.Tensor(motion_coords[img_i]).cuda()
        mask_gt     = masks[img_i].cuda()

        # ── Optical Flow Loading ───────────────────────────────────────────────
        if args.semantic_flow:
            # Trajectory-based flow from DINO-tracker
            if img_i == 0:
                flow_fwd, fwd_mask = read_semantic_flow(args.datadir, img_i, fwd=True,
                                                    start_frame=args.start_frame, step=args.step_frame)
                flow_bwd = np.zeros_like(flow_fwd)
                bwd_mask = np.zeros_like(fwd_mask)
            elif img_i == num_img - 1:
                flow_bwd, bwd_mask = read_semantic_flow(args.datadir, img_i, fwd=False,
                                                    start_frame=args.start_frame, step=args.step_frame)
                flow_fwd = np.zeros_like(flow_bwd)
                fwd_mask = np.zeros_like(bwd_mask)
            else:
                flow_fwd, fwd_mask = read_semantic_flow(args.datadir, img_i, fwd=True,
                                                    start_frame=args.start_frame, step=args.step_frame)
                flow_bwd, bwd_mask = read_semantic_flow(args.datadir, img_i, fwd=False,
                                                    start_frame=args.start_frame, step=args.step_frame)
        else:
            # Standard RAFT optical flow
            if img_i == 0:
                flow_fwd, fwd_mask = read_optical_flow(args.datadir, img_i, args.start_frame,
                                                        fwd=True,  step=args.step_frame)
                flow_bwd = np.zeros_like(flow_fwd)
                bwd_mask = np.zeros_like(fwd_mask)
            elif img_i == num_img - 1:
                flow_bwd, bwd_mask = read_optical_flow(args.datadir, img_i, args.start_frame,
                                                        fwd=False, step=args.step_frame)
                flow_fwd = np.zeros_like(flow_bwd)
                fwd_mask = np.zeros_like(bwd_mask)
            else:
                flow_fwd, fwd_mask = read_optical_flow(args.datadir, img_i, args.start_frame,
                                                        fwd=True,  step=args.step_frame)
                flow_bwd, bwd_mask = read_optical_flow(args.datadir, img_i, args.start_frame,
                                                        fwd=False, step=args.step_frame)

        # Convert flow to absolute pixel coordinates (flow + grid = target location)
        flow_fwd = torch.Tensor(flow_fwd).cuda() + uv_grid
        fwd_mask = torch.Tensor(fwd_mask).cuda()
        flow_bwd = torch.Tensor(flow_bwd).cuda() + uv_grid
        bwd_mask = torch.Tensor(bwd_mask).cuda()

        # ── Ray Batch Construction ─────────────────────────────────────────────
        rays_o, rays_d = get_rays(H, W, focal, torch.Tensor(pose))
        coords = torch.stack(
            torch.meshgrid(torch.linspace(0, H - 1, H),
                           torch.linspace(0, W - 1, W)), -1
        ).reshape(-1, 2)  # (H*W, 2)

        if args.use_difix and i > difix_start_step and is_syn:
            # Difix step: patch-based or motion-weighted sampling on synthetic view
            if args.use_percept:
                coords_np = np.asarray(difix_updater.motion_coords[img_i][which_syn], dtype=np.int64)
                select_coords = sample_mixed_patches(coords_np, H, W, patch_sz, 1, 1.0, device)
            else:
                difix_coords = torch.Tensor(difix_updater.motion_coords[img_i][which_syn]).cuda()
                inds = np.random.choice(difix_coords.shape[0], size=[N_rand], replace=False)
                select_coords = difix_coords[inds].long()
        elif args.use_motion_mask and i < decay_iteration * 1000:
            # Hard mining: over-sample motion-region pixels during early training
            inds_hard = np.random.choice(hard_coords.shape[0],
                                         size=[min(hard_coords.shape[0], args.num_extra_sample)],
                                         replace=False)
            inds_all  = np.random.choice(coords.shape[0], size=[N_rand], replace=False)
            select_coords = torch.cat([coords[inds_all].long(),
                                       hard_coords[inds_hard].long()], 0)
        else:
            inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)
            select_coords = coords[inds].long()

        rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]
        rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]
        batch_rays    = torch.stack([rays_o, rays_d], 0)
        target_rgb    = target[select_coords[:, 0], select_coords[:, 1]]
        target_depth  = depth_gt[select_coords[:, 0], select_coords[:, 1]]
        target_mask   = mask_gt[select_coords[:, 0], select_coords[:, 1]].unsqueeze(-1)
        target_of_fwd  = flow_fwd[select_coords[:, 0], select_coords[:, 1]]
        target_fwd_mask = fwd_mask[select_coords[:, 0], select_coords[:, 1]].unsqueeze(-1)
        target_of_bwd  = flow_bwd[select_coords[:, 0], select_coords[:, 1]]
        target_bwd_mask = bwd_mask[select_coords[:, 0], select_coords[:, 1]].unsqueeze(-1)

        # Temporal embedding: normalized frame index in [-1, 1]
        if args.sparse:
            step_of_pose = (poses[:-1, :, 3] - poses[1:, :, 3]).pow(2).sum(1).sqrt()
            step_of_pose = torch.cat((torch.tensor([0.0], device=poses.device), step_of_pose))
            pose_length  = step_of_pose.sum()
            img_idx_embed = step_of_pose[:img_i + 1].sum() / pose_length * 2. - 1.0
        else:
            img_idx_embed = img_i / num_img * 2. - 1.0

        # ── Scene Flow Chain Scheduling ────────────────────────────────────────
        # After 2× decay_iteration epochs, enable 5-frame chain scene-flow loss
        if args.chain_sf and i > decay_iteration * 1000 * 2:
            chain_5frames = True
        else:
            chain_5frames = False

        # ── Forward Render ─────────────────────────────────────────────────────
        optimizer.zero_grad()
        ret = render(img_idx_embed, chain_bwd, chain_5frames,
                     num_img, H, W, focal, chunk=args.chunk,
                     rays=batch_rays, verbose=i < 10, retraw=True,
                     **render_kwargs_train)

        # Refresh Difix synthetic views at fixed intervals
        if args.use_difix and i >= difix_start_step:
            if global_step >= next_refresh:
                difix_updater.update(global_step, difix_pipe, images, Cam_param,
                                     True, args.difix_ref_scale)
                next_refresh  += args.difix_interval
                images_syn = difix_updater.synthetic_imgs.to(images.device)
                poses_syn  = difix_updater.synthetic_poses.to(poses.device)
                images = torch.cat([images_orig, images_syn], 0)
                poses  = torch.cat([poses_orig,  poses_syn],  0)

        # ── Tone Mapping (HDR radiance → LDR via CRF) ─────────────────────────
        if args.use_tone_mapping:
            rgb_map_ref     = Cam_param.RAD2LDR(ret['rgb_map_ref'],     img_i, is_train=True)
            rgb_map_rig     = Cam_param.RAD2LDR(ret['rgb_map_rig'],     img_i, is_train=True)
            rgb_map_ref_dy  = Cam_param.RAD2LDR(ret['rgb_map_ref_dy'],  img_i, is_train=True)
            rgb_map_prev_dy = Cam_param.RAD2LDR(ret['rgb_map_prev_dy'], img_i, is_train=True)
            rgb_map_post_dy = Cam_param.RAD2LDR(ret['rgb_map_post_dy'], img_i, is_train=True)
            if chain_5frames:
                rgb_map_pp_dy = Cam_param.RAD2LDR(ret['rgb_map_pp_dy'], img_i, is_train=True)
        else:
            rgb_map_ref     = ret['rgb_map_ref']
            rgb_map_rig     = ret['rgb_map_rig']
            rgb_map_ref_dy  = ret['rgb_map_ref_dy']
            rgb_map_prev_dy = ret['rgb_map_prev_dy']
            rgb_map_post_dy = ret['rgb_map_post_dy']
            if chain_5frames:
                rgb_map_pp_dy = ret['rgb_map_pp_dy']

        # ── Rendered Optical Flow (projected from 3D scene flow) ───────────────
        pose_post = poses[min(img_i + 1, int(num_img) - 1), :3, :4]
        pose_prev = poses[max(img_i - 1, 0),                :3, :4]
        if args.use_difix and i > difix_start_step:
            pose = poses[img_i]
        render_of_fwd, render_of_bwd = compute_optical_flow(
            pose_post, pose, pose_prev, H, W, focal, ret)

        # ── CamParam Gradient Zeroing ──────────────────────────────────────────
        if args.use_tone_mapping:
            optim_cam_wb.zero_grad(set_to_none=True)
            if optim_cam_crf is not None:
                optim_cam_crf.zero_grad(set_to_none=True)

        # Permanently freeze CamParam once CRF training phase ends
        if (args.use_tone_mapping and not Cam_param.is_frozen_forever()
                and global_step > crf_stop_step):
            print(f'[Step {global_step}] Freezing CamParam permanently.')
            Cam_param.freeze_forever()

        # ── Loss Computation ───────────────────────────────────────────────────
        weight_map_post = ret['prob_map_post']
        weight_map_prev = ret['prob_map_prev']
        weight_post = 1. - ret['raw_prob_ref2post']
        weight_prev = 1. - ret['raw_prob_ref2prev']

        # Saturation mask: de-emphasize clipped (over/under-exposed) pixels
        if args.use_sat_mask:
            sat_mask = saturation_mask(target_rgb,
                                       threshold_low=args.leaky_th_l,
                                       threshold_high=args.leaky_th_h)

        prob_reg_loss = args.w_prob_reg * (torch.mean(torch.abs(ret['raw_prob_ref2prev']))
                                           + torch.mean(torch.abs(ret['raw_prob_ref2post'])))

        # Helper: dispatch to the appropriate photometric loss variant
        def _render_loss(pred, tgt, weight=None):
            if args.use_sat_mask:
                if args.use_render_l1 or l1_flag:
                    return compute_mae_sat(pred, tgt, weight, sat_mask)
                else:
                    return compute_mse_sat(pred, tgt, weight, sat_mask)
            else:
                if args.use_render_l1 or l1_flag:
                    return compute_mae(pred, tgt, weight) if weight is not None else img2mae(pred, tgt)
                else:
                    return compute_mse(pred, tgt, weight) if weight is not None else img2mse(pred, tgt)

        # Photometric loss on dynamic branch
        if i <= decay_iteration * 1000:
            # Early training: all rays weighted equally
            ones = torch.ones_like(target_rgb[..., :1])
            render_loss  = _render_loss(rgb_map_ref_dy,  target_rgb, ones)
            render_loss += _render_loss(rgb_map_post_dy, target_rgb, weight_map_post.unsqueeze(-1))
            render_loss += _render_loss(rgb_map_prev_dy, target_rgb, weight_map_prev.unsqueeze(-1))
        else:
            # Late training: weight by predicted dynamic probability map
            weights_map_dd = ret['weights_map_dd'].unsqueeze(-1).detach()
            render_loss  = _render_loss(rgb_map_ref_dy,  target_rgb, weights_map_dd)
            render_loss += _render_loss(rgb_map_post_dy, target_rgb,
                                        weight_map_post.unsqueeze(-1) * weights_map_dd)
            render_loss += _render_loss(rgb_map_prev_dy, target_rgb,
                                        weight_map_prev.unsqueeze(-1) * weights_map_dd)

        # Photometric loss on composite (static + dynamic) render
        if args.use_render_l1 or l1_flag:
            render_loss += img2mae(rgb_map_ref[:N_rand], target_rgb[:N_rand])
        else:
            render_loss += img2mse(rgb_map_ref[:N_rand], target_rgb[:N_rand])

        # Perceptual loss on patch-sampled difix synthetic views
        if args.use_difix and i > difix_start_step and is_syn and args.use_percept:
            def _patch_feats(img):
                return maxpool_feats(perceptual_net(
                    img.reshape(patch_sz, patch_sz, 3).permute(2, 0, 1).unsqueeze(0)))
            with torch.no_grad():
                feat_tgt = _patch_feats(target_rgb)
            percept_loss = args.w_percept * (
                F.l1_loss(_patch_feats(rgb_map_ref),     feat_tgt)
                + F.l1_loss(_patch_feats(rgb_map_post_dy), feat_tgt)
                + F.l1_loss(_patch_feats(rgb_map_prev_dy), feat_tgt)
            ) / 3.
            render_loss = 0.  # Replace photometric loss with perceptual in difix step
        else:
            percept_loss = None

        # Optional 5-frame chain render loss
        if chain_5frames:
            w_5f = weights_map_dd if i > decay_iteration * 1000 else torch.ones_like(target_rgb[..., :1])
            render_loss += _render_loss(rgb_map_pp_dy, target_rgb, w_5f)

        # ── Scene Flow Losses ──────────────────────────────────────────────────
        sf_cycle_loss = args.w_cycle * (
            compute_mae(ret['raw_sf_ref2post'], -ret['raw_sf_post2ref'],
                        weight_post.unsqueeze(-1), dim=3)
            + compute_mae(ret['raw_sf_ref2prev'], -ret['raw_sf_prev2ref'],
                          weight_prev.unsqueeze(-1), dim=3))

        # Scene flow magnitude regularization (encourage small flow)
        render_sf_ref2prev = torch.sum(
            ret['weights_ref_dy'].unsqueeze(-1) * ret['raw_sf_ref2prev'], -1)
        render_sf_ref2post = torch.sum(
            ret['weights_ref_dy'].unsqueeze(-1) * ret['raw_sf_ref2post'], -1)
        sf_reg_loss = args.w_sf_reg * (torch.mean(torch.abs(render_sf_ref2prev))
                                       + torch.mean(torch.abs(render_sf_ref2post)))

        # Scene flow smoothness + least kinetic energy losses
        sf_sm_loss = args.w_sm * (
            compute_sf_sm_loss(ret['raw_pts_ref'], ret['raw_pts_post'], H, W, focal)
            + compute_sf_sm_loss(ret['raw_pts_ref'], ret['raw_pts_prev'], H, W, focal)
            + 2 * compute_sf_lke_loss(ret['raw_pts_ref'], ret['raw_pts_post'],
                                      ret['raw_pts_prev'], H, W, focal))
        # Two-frame chain kinetic energy (alternates fwd/bwd each iteration)
        if chain_bwd:
            sf_sm_loss += args.w_sm * compute_sf_lke_loss(
                ret['raw_pts_prev'], ret['raw_pts_ref'], ret['raw_pts_pp'], H, W, focal)
        else:
            sf_sm_loss += args.w_sm * compute_sf_lke_loss(
                ret['raw_pts_post'], ret['raw_pts_pp'], ret['raw_pts_ref'], H, W, focal)

        entropy_loss = args.w_entropy * torch.mean(
            -ret['raw_blend_w'] * torch.log(ret['raw_blend_w'] + 1e-8))

        # ── Loss Weight Decay ──────────────────────────────────────────────────
        divsor = i // (decay_iteration * 1000)
        decay_factor = 10
        w_depth = (args.w_depth / (decay_factor ** divsor)
                   if args.decay_depth_w else args.w_depth)
        w_of    = (args.w_optical_flow / (decay_factor ** divsor)
                   if args.decay_optical_flow_w else args.w_optical_flow)

        # Suppress depth / flow losses on difix synthetic views
        if args.use_difix and i > difix_start_step and is_syn:
            w_depth = 0.
            w_of = 0.

        w_crf_smooth = (args.w_crf_smooth / (decay_factor ** divsor)
                        if args.decay_crf_w else args.w_crf_smooth)

        # ── Depth Loss ─────────────────────────────────────────────────────────
        depth_loss = w_depth * compute_depth_loss(
            ret['depth_map_ref_dy'], -1. / (target_depth + 1e-5))

        # ── Optical Flow Loss ──────────────────────────────────────────────────
        if img_i == 0:
            flow_loss = w_of * compute_mae(render_of_fwd, target_of_fwd, target_fwd_mask)
        elif img_i == num_img - 1:
            flow_loss = w_of * compute_mae(render_of_bwd, target_of_bwd, target_bwd_mask)
        else:
            flow_loss = (w_of * compute_mae(render_of_fwd, target_of_fwd, target_fwd_mask)
                         + w_of * compute_mae(render_of_bwd, target_of_bwd, target_bwd_mask))

        # Optional render loss scaling
        if args.w_render != 1.:
            render_loss = args.w_render * render_loss

        # ── Total Loss ─────────────────────────────────────────────────────────
        loss = (render_loss + flow_loss + depth_loss
                + sf_reg_loss + sf_cycle_loss + sf_sm_loss
                + prob_reg_loss + entropy_loss)

        # CRF regularization (active only while CRF is still being trained)
        if args.use_tone_mapping and global_step <= crf_stop_step and not camparam_frozen:
            if optim_cam_crf is not None:
                crf_smooth_loss = w_crf_smooth * Cam_param.crf_smoothness_loss()
                loss = loss + crf_smooth_loss
            else:
                crf_smooth_loss = None
        else:
            crf_smooth_loss = None

        if percept_loss is not None:
            loss = loss + percept_loss

        # ── Backprop & Optimizer Step ──────────────────────────────────────────
        loss.backward()
        optimizer.step()

        if args.use_tone_mapping and not freeze_cam_this_iter and not Cam_param.is_frozen_forever():
            optim_cam_wb.step()
            if optim_cam_crf is not None:
                optim_cam_crf.step()

        # ── Step Logging ───────────────────────────────────────────────────────
        dt = time.time() - time0
        rl  = render_loss.item() if isinstance(render_loss, torch.Tensor) else render_loss
        print(f'Step {global_step:6d} | loss {loss.item():.4f} | render {rl:.4f} | '
              f'depth {depth_loss.item():.4f} | flow {flow_loss.item():.4f} | dt {dt:.2f}s')

        if i % 100 == 0 and args.use_tone_mapping and Cam_param is not None:
            print(f'[Iter {i}] is_syn={int(use_difix_flag and is_syn != 0)} '
                  f'WB_grad={grad_on(Cam_param.wb)} CRF_grad={grad_on(Cam_param.CRF)} '
                  f'frozen_iter={Cam_param.is_frozen_this_iter()} '
                  f'frozen_forever={Cam_param.is_frozen_forever()}')

        if not debug_mode:
            log_dict = {
                'loss':             loss.item(),
                'render_loss':      rl,
                'bidirection_loss': sf_cycle_loss.item(),
                'sf_reg_loss':      sf_reg_loss.item(),
                'depth_loss':       depth_loss.item(),
                ('semantic_flow_loss' if args.semantic_flow else 'flow_loss'): flow_loss.item(),
                'sf_sm_loss':       sf_sm_loss.item(),
                'prob_reg_loss':    prob_reg_loss.item(),
                'entropy_loss':     entropy_loss.item(),
            }
            if crf_smooth_loss is not None: log_dict['crf_smoothness_loss'] = crf_smooth_loss.item()
            if percept_loss    is not None: log_dict['perceptual_loss']     = percept_loss.item()
            wandb.log(log_dict, step=i)

        # ── Learning Rate Decay ────────────────────────────────────────────────
        if args.stage == 2:
            new_lrate = 1e-3
        else:
            decay_steps = args.lrate_decay * 1000
            new_lrate   = args.lrate * (0.1 ** (global_step / decay_steps))
            for pg in optimizer.param_groups:
                pg['lr'] = new_lrate
            if args.use_tone_mapping and not Cam_param.is_frozen_forever():
                if optim_cam_crf is not None:
                    for pg in optim_cam_crf.param_groups:
                        pg['lr'] = (new_lrate * args.share_crf_lr_ratio
                                    if args.share_crf else new_lrate)
                for pg in optim_cam_wb.param_groups:
                    pg['lr'] = new_lrate

        # Lazily initialize CRF optimizer after warm-up steps
        if args.use_tone_mapping and args.tone_mapping == 'piece_wise':
            if optim_cam_crf is None and global_step > args.crf_start_step:
                print(f'[Step {global_step}] Adding CRF optimizer.')
                lr_crf = new_lrate / num_img if args.share_crf else new_lrate
                optim_cam_crf = torch.optim.Adam(
                    params=Cam_param.crf_params, lr=lr_crf, betas=(0.9, 0.999))

        # ── Checkpoint ────────────────────────────────────────────────────────
        if i % args.i_weights == 0:
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
            torch.save({
                'global_step':             global_step,
                'network_fn_state_dict':   render_kwargs_train['network_fn'].state_dict(),
                'network_rigid':           render_kwargs_train['network_rigid'].state_dict(),
                'optimizer_state_dict':    optimizer.state_dict(),
                'optim_cam_crf_state_dict': optim_cam_crf.state_dict() if optim_cam_crf is not None else None,
                'optim_cam_wb_state_dict':  optim_cam_wb.state_dict()  if optim_cam_wb  is not None else None,
                'white_balance': Cam_param.wb.state_dict()  if Cam_param is not None else None,
                'CRF':           (Cam_param.CRF.state_dict()
                                  if Cam_param is not None and Cam_param.CRF is not None else None),
                'rgb_model':     (Cam_param.rgb_model.state_dict()
                                  if Cam_param is not None and args.tone_mapping == 'nn' else None),
                'color_refiner': (render_kwargs_train['color_refiner'].state_dict()
                                  if args.stage == 2 else None),
            }, path)
            print('Saved checkpoint:', path)

        # ── Periodic Validation Visualization ─────────────────────────────────
        if i % args.i_print == 0 and i > 0:
            writer.add_scalar("train/loss",          loss.item(),          i)
            writer.add_scalar("train/render_loss",   rl,                   i)
            writer.add_scalar("train/depth_loss",    depth_loss.item(),    i)
            writer.add_scalar("train/flow_loss",     flow_loss.item(),     i)
            writer.add_scalar("train/prob_reg_loss", prob_reg_loss.item(), i)
            writer.add_scalar("train/sf_reg_loss",   sf_reg_loss.item(),   i)
            writer.add_scalar("train/sf_cycle_loss", sf_cycle_loss.item(), i)
            writer.add_scalar("train/sf_sm_loss",    sf_sm_loss.item(),    i)

        if i % args.i_img == 0 and not debug_mode:
            img_idx_embed_val = img_i / num_img * 2. - 1.0
            target_val       = images[img_i]
            pose_val         = poses[img_i, :3, :4]
            target_depth_val = depths[img_i] - torch.min(depths[img_i])
            with torch.no_grad():
                ret_v = render(img_idx_embed_val, chain_bwd, False,
                               num_img, H, W, focal, chunk=1024 * 16,
                               c2w=pose_val, **render_kwargs_test)

            if args.use_tone_mapping:
                ldr_img = torch.clamp(Cam_param.RAD2LDR_img(ret_v['rgb_map_ref'], img_i), 0., 1.)
                mu50    = torch.log1p(ret_v['rgb_map_ref'] * 50) / torch.log1p(torch.tensor(50., device=device))
                wandb.log({
                    "rgb_map_ref_ldr": wandb.Image((ldr_img.detach() * 255).cpu().numpy().astype(np.uint8)),
                    "rgb_map_ref":     wandb.Image((torch.clamp(mu50, 0., 1.).detach() * 255).cpu().numpy().astype(np.uint8)),
                    "wb":              wandb.Image(Cam_param.get_wb_plot()),
                }, step=i)
                if args.share_crf:
                    wandb.log({"share_crf": wandb.Image(Cam_param.get_visualize_crf(0))}, step=i)
                elif args.tone_mapping == 'piece_wise':
                    wandb.log({"crf": wandb.Image(Cam_param.get_visualize_crf(img_i))}, step=i)
                elif args.tone_mapping == 'hdr_hexplane':
                    wandb.log({"crf": wandb.Image(Cam_param.get_visualize_gamma())}, step=i)
                elif args.tone_mapping == 'nn':
                    wandb.log({"crf": wandb.Image(Cam_param.get_visualize_crf_nn())}, step=i)
                mse  = torch.mean((ldr_img - target_val.to(ldr_img.device)) ** 2)
                wandb.log({"psnr": (-10. * torch.log10(mse + 1e-8)).item()}, step=i)
            else:
                rgb_ref = torch.clamp(ret_v['rgb_map_ref'], 0., 1.)
                wandb.log({"rgb_map_ref": wandb.Image(
                    (rgb_ref * 255).cpu().numpy().astype(np.uint8))}, step=i)
                mse  = torch.mean((ret_v['rgb_map_ref'] - target_val.to(ret_v['rgb_map_ref'].device)) ** 2)
                wandb.log({"psnr": (-10. * torch.log10(mse + 1e-8)).item()}, step=i)

            def _depth_colormap(t):
                return cv2.applyColorMap(
                    (normalize_depth(t).cpu().numpy() * 255).astype(np.uint8),
                    cv2.COLORMAP_PLASMA)

            wandb.log({
                "depth_map_ref":    wandb.Image(_depth_colormap(ret_v['depth_map_ref'])),
                "rgb_map_rig":      wandb.Image((torch.clamp(ret_v['rgb_map_rig'], 0., 1.) * 255).cpu().numpy().astype(np.uint8)),
                "depth_map_rig":    wandb.Image(_depth_colormap(ret_v['depth_map_rig'])),
                "rgb_map_ref_dy":   wandb.Image((torch.clamp(ret_v['rgb_map_ref_dy'], 0., 1.) * 255).cpu().numpy().astype(np.uint8)),
                "depth_map_ref_dy": wandb.Image(_depth_colormap(ret_v['depth_map_ref_dy'])),
                "gt_rgb":           wandb.Image((torch.clamp(target_val, 0., 1.) * 255).cpu().numpy().astype(np.uint8)),
                "weights_map_dd":   wandb.Image(ret_v['weights_map_dd'].cpu().numpy()),
            }, step=i)

            mono_disp = cv2.applyColorMap(
                (torch.clamp(target_depth_val / percentile(target_depth_val, 97), 0., 1.)
                 .cpu().numpy() * 255).astype(np.uint8),
                cv2.COLORMAP_PLASMA)
            wandb.log({"monocular_disp": wandb.Image(mono_disp)}, step=i)


            writer.add_image("val/rgb_map_ref",      torch.clamp(ret_v['rgb_map_ref'],    0., 1.), i, dataformats='HWC')
            writer.add_image("val/depth_map_ref",    normalize_depth(ret_v['depth_map_ref']),      i, dataformats='HW')
            writer.add_image("val/rgb_map_rig",      torch.clamp(ret_v['rgb_map_rig'],    0., 1.), i, dataformats='HWC')
            writer.add_image("val/depth_map_rig",    normalize_depth(ret_v['depth_map_rig']),      i, dataformats='HW')
            writer.add_image("val/rgb_map_ref_dy",   torch.clamp(ret_v['rgb_map_ref_dy'], 0., 1.), i, dataformats='HWC')
            writer.add_image("val/depth_map_ref_dy", normalize_depth(ret_v['depth_map_ref_dy']),   i, dataformats='HW')
            writer.add_image("val/gt_rgb",           target_val,                                    i, dataformats='HWC')
            writer.add_image("val/monocular_disp",
                             torch.clamp(target_depth_val / percentile(target_depth_val, 97), 0., 1.),
                             i, dataformats='HW')
            writer.add_image("val/weights_map_dd",   ret_v['weights_map_dd'],                      i, dataformats='HW')

        global_step += 1

    if not debug_mode:
        wandb.finish()


if __name__ == '__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    train()
