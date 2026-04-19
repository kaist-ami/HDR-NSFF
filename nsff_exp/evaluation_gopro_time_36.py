import os, sys
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import numpy as np
import time
import torch
import cv2

from render_utils import *
from run_nerf_helpers import *
from load_llff import load_nvidia_data
import cam_param
from config import config_parser

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(1)


def evaluation():

    parser = config_parser()
    args = parser.parse_args()

    # ── Data Loading ───────────────────────────────────────────────────────────
    if args.dataset_type == 'llff':
        target_idx = args.target_idx
        images, poses, bds, render_poses = load_nvidia_data(
            args.datadir,
            args.start_frame, args.end_frame,
            args.factor,
            target_idx=target_idx,
            recenter=True, bd_factor=.9,
            spherify=args.spherify,
            final_height=args.final_height,
        )

        hwf = poses[0, :3, -1]
        poses = poses[:, :3, :4]
        print('Loaded llff', images.shape, render_poses.shape, hwf, args.datadir)

        i_test  = []
        i_val   = []
        i_train = np.array([i for i in np.arange(int(images.shape[0]))
                            if i not in i_test and i not in i_val])

        if args.no_ndc:
            near = np.percentile(bds[:, 0], 5) * 0.9
            far  = np.percentile(bds[:, 1], 95) * 1.1
        else:
            near, far = 0., 1.
        print('NEAR FAR', near, far)

    else:
        print('Unsupported dataset type:', args.dataset_type)
        sys.exit()

    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    # ── Experiment Directory ───────────────────────────────────────────────────
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

    bds_dict = {'near': near, 'far': far}
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    # ── Camera Parameters (HDR / CRF) ─────────────────────────────────────────
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
    else:
        Cam_param = None

    # ── Evaluation ────────────────────────────────────────────────────────────
    num_img = float(images.shape[0])
    poses   = torch.Tensor(poses).to(device)

    # Determine output directory name (use ft_path suffix when fine-tuning)
    if args.ft_path is not None and args.ft_path != 'None':
        eval_dir = 'evaluation_' + args.ft_path.split('/')[-2] + '_' + os.path.basename(args.ft_path)
    else:
        eval_dir = 'evaluation_time_36'
    evaluation_render_dir = os.path.join(basedir, expname, eval_dir)
    os.makedirs(evaluation_render_dir, exist_ok=True)

    with torch.no_grad():
        t = time.time()
        for img_i in i_train:
            # Temporal blend ratio between frame img_i and img_i+1
            ratio = 0.5
            img_idx_embed_1 = img_i / num_img * 2. - 1.0
            img_idx_embed_2 = (img_i + 1) / num_img * 2. - 1.0

            pred_ldr_dir     = os.path.join(evaluation_render_dir, 'ldr',     f'{img_i:05d}')
            pred_hdr_exr_dir = os.path.join(evaluation_render_dir, 'hdr_exr', f'{img_i:05d}')
            pred_hdr_tm_dir  = os.path.join(evaluation_render_dir, 'hdr_tm',  f'{img_i:05d}')
            os.makedirs(pred_ldr_dir,     exist_ok=True)
            os.makedirs(pred_hdr_exr_dir, exist_ok=True)
            os.makedirs(pred_hdr_tm_dir,  exist_ok=True)

            # Render all 9 synchronized cameras for this time step
            for camera_i in range(9):
                print(f'frame {img_i}  camera {camera_i}  ({time.time() - t:.1f}s)')
                t = time.time()

                render_pose = poses[camera_i]
                R_w2t = render_pose[:3, :3].transpose(0, 1)
                t_w2t = -torch.matmul(R_w2t, render_pose[:3, 3:4])

                # Render at both adjacent time steps for temporal splatting
                ret1 = render_sm(img_idx_embed_1, 0, False,
                                 num_img, H, W, focal,
                                 chunk=1024 * 8, c2w=render_pose,
                                 **render_kwargs_test)
                ret2 = render_sm(img_idx_embed_2, 0, False,
                                 num_img, H, W, focal,
                                 chunk=1024 * 8, c2w=render_pose,
                                 **render_kwargs_test)

                # Alpha-composite splatted samples from both time steps
                T_i       = torch.ones((1, H, W))
                final_rgb = torch.zeros((3, H, W))
                num_sample = ret1['raw_rgb'].shape[2]

                for j in range(num_sample):
                    splat_alpha_dy_1, splat_rgb_dy_1, \
                    splat_alpha_rig_1, splat_rgb_rig_1 = splat_rgb_img(
                        ret1, ratio, R_w2t, t_w2t, j, H, W, focal, True)
                    splat_alpha_dy_2, splat_rgb_dy_2, \
                    splat_alpha_rig_2, splat_rgb_rig_2 = splat_rgb_img(
                        ret2, 1. - ratio, R_w2t, t_w2t, j, H, W, focal, False)

                    final_rgb += T_i * (splat_alpha_dy_1 * splat_rgb_dy_1 +
                                        splat_alpha_rig_1 * splat_rgb_rig_1) * (1.0 - ratio)
                    final_rgb += T_i * (splat_alpha_dy_2 * splat_rgb_dy_2 +
                                        splat_alpha_rig_2 * splat_rgb_rig_2) * ratio

                    alpha_1_final = (1.0 - (1. - splat_alpha_dy_1) * (1. - splat_alpha_rig_1)) * (1. - ratio)
                    alpha_2_final = (1.0 - (1. - splat_alpha_dy_2) * (1. - splat_alpha_rig_2)) * ratio
                    alpha_final   = alpha_1_final + alpha_2_final
                    T_i = T_i * (1.0 - alpha_final + 1e-10)

                if Cam_param is not None:
                    # Save raw HDR radiance as EXR (OpenCV expects BGR)
                    cv2.imwrite(
                        os.path.join(pred_hdr_exr_dir, f'{camera_i + 1:05d}.exr'),
                        final_rgb.permute(1, 2, 0).cpu().numpy().astype(np.float32)[..., ::-1])

                    # Mu-law tone-mapped preview
                    mu          = torch.tensor(50, dtype=final_rgb.dtype, device=final_rgb.device)
                    final_rgb_tm = torch.log(1 + mu * final_rgb) / torch.log(1 + mu)
                    imageio.imwrite(
                        os.path.join(pred_hdr_tm_dir, f'{camera_i + 1:05d}.png'),
                        to8b(final_rgb_tm.permute(1, 2, 0).cpu().numpy()))

                    # Per-exposure LDR via learned CRF
                    rgb8_ldr = to8b(Cam_param.RAD2LDR_img(
                        final_rgb.permute(1, 2, 0), camera_i).cpu().numpy())
                else:
                    rgb8_ldr = to8b(final_rgb.permute(1, 2, 0).cpu().numpy())

                imageio.imwrite(
                    os.path.join(pred_ldr_dir, f'{camera_i + 1:05d}.png'),
                    rgb8_ldr)


if __name__ == '__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    evaluation()
