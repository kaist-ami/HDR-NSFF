import os, sys
import numpy as np
import time
import torch
import math
import models
import cv2

from render_utils import *
from run_nerf_helpers import *
from config import config_parser
import cam_param

from load_llff import load_nvidia_data
from load_llff import load_hdr_blender_data
from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio


def im2tensor(image, imtype=np.uint8, cent=1., factor=1./2.):
    return torch.Tensor((image / factor - cent)
                        [:, :, :, np.newaxis].transpose((3, 2, 0, 1)))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(1)
DEBUG = False


def calculate_psnr(img1, img2, mask):
    # img1 and img2 have range [0, 1]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mask = mask.astype(np.float64)

    num_valid = np.sum(mask) + 1e-8

    mse = np.sum((img1 - img2)**2 * mask) / num_valid
    
    if mse == 0:
        return 0 #float('inf')

    return 10 * math.log10(1./mse)


def calculate_ssim(img1, img2, mask):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 1]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')

    _, ssim_map = structural_similarity(img1, img2, multichannel=True, full=True)
    num_valid = np.sum(mask) + 1e-8

    return np.sum(ssim_map * mask) / num_valid



def evaluation():
    
    parser = config_parser()
    args = parser.parse_args()

    # Load data
    if args.dataset_type == 'llff':
        target_idx = args.target_idx
        images, poses, bds, render_poses = load_nvidia_data(args.datadir, 
                                                            args.start_frame, args.end_frame,
                                                            args.factor,
                                                            target_idx=target_idx,
                                                            recenter=True, bd_factor=.9,
                                                            spherify=args.spherify, 
                                                            final_height=args.final_height)


        hwf = poses[0,:3,-1]
        poses = poses[:,:3,:4]
        print('Loaded llff', images.shape, render_poses.shape, hwf, args.datadir)
        # if not isinstance(i_test, list):
        i_test = []
        i_val = [] #i_test
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                        (i not in i_test and i not in i_val)])

        print('DEFINING BOUNDS')
        if args.no_ndc:
            near = np.percentile(bds[:, 0], 5) * 0.9 #np.ndarray.min(bds) #* .9
            far = np.percentile(bds[:, 1], 95) * 1.1 #np.ndarray.max(bds) #* 1.
        else:
            near = 0.
            far = 1.

        print('NEAR FAR', near, far)
    
    elif args.dataset_type == 'hdr-blender':
        
        target_idx = args.target_idx
        images, poses, bds, render_poses = load_hdr_blender_data(args.datadir,
                                                                 args.start_frame, args.end_frame, args.step_frame,
                                                                 args.factor,
                                                                 target_idx=target_idx,
                                                                 recenter=True, bd_factor=.9,
                                                                 spherify=args.spherify,
                                                                 final_height=args.final_height)
          
        hwf = poses[0,:3,-1]
        poses = poses[:,:3,:4]
        print('Loaded hdr-blender', images.shape, render_poses.shape, hwf, args.datadir)
        
        i_test = []
        i_val = []
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if (i not in i_test and i not in i_val)])
        
        print('DEFINING BOUNDS')
        if args.no_ndc:
            near = np.percentile(bds[:, 0], 5) * 0.9 #np.ndarray.min(bds) #* .9
            far = np.percentile(bds[:, 1], 95) * 1.1 #np.ndarray.max(bds) #* 1.
        else:
            near = 0.
            far = 1.

        print('NEAR FAR', near, far)
    
    else:
        print('ONLY SUPPORT LLFF!!!!!!!!')
        sys.exit()


    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    # Create log dir and copy the config file
    basedir = args.basedir
    args.expname = args.expname + '_F%02d-%02d'%(args.start_frame, 
                                                 args.end_frame)
    expname = args.expname

    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    # Create nerf model
    render_kwargs_train, render_kwargs_test, \
        start, grad_vars, optimizer = create_nerf(args)

    global_step = start

    bds_dict = {
        'near' : near,
        'far' : far,
    }

    if args.use_tone_mapping == True:
        # #FIXME: 이 부분 고쳐야함!
        N, H, W, C = images.shape
        Cam_param = cam_param.CamParam(N, H, W, device=device, gts=images, initialize=args.use_initialize, tone_mapping=args.tone_mapping, share_crf=args.share_crf, share_wb=args.share_wb, ref_idx=args.ref_idx, log_scale=args.log_scale)
        # optim_cam = Cam_param.optimizer(l_rate = args.lrate)
        optim_cam_wb, optim_cam_crf = Cam_param.optimizer(l_rate = args.lrate)

        basedir = args.basedir
        expname = args.expname
        if args.ft_path is not None and args.ft_path!='None':
            ckpts = [args.ft_path]
        else:
            ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if 'tar' in f]

        if len(ckpts) > 0 and not args.no_reload:
            ckpt_path = ckpts[-1]
            
            print('Reloading CamParam from', ckpt_path)
            ckpt = torch.load(ckpt_path)
            Cam_param.load_ckpt(ckpt)
    else:
        Cam_param = None
    
    # if args.use_tone_mapping_camparam == True:
    #     Cam_param, optim_cam = create_CamParam(args, device, images, test=True)
    # else:
    #     Cam_param = None
    #     optim_cam = None

    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)
    num_img = float(images.shape[0])
    poses = torch.Tensor(poses).to(device)

    with torch.no_grad():

        model = models.PerceptualLoss(model='net-lin',net='alex',
                                      use_gpu=True,version=0.1)

        total_psnr = 0.
        total_ssim = 0.
        total_lpips = 0.
        count = 0
        total_psnr_dy = 0.
        total_ssim_dy = 0.
        total_lpips_dy = 0.
        t = time.time()
        
        mid = 0
        high = 2
        low = 1
        
        evaluation_render_dir = os.path.join(basedir, expname, 'evaluation_time')
        # evaluation_gt_dir = os.path.join(evaluation_render_dir, 'gt')
        os.makedirs(evaluation_render_dir, exist_ok=True)
        # os.makedirs(evaluation_gt_dir, exist_ok=True)
        os.makedirs(os.path.join(evaluation_render_dir, 'hdr'), exist_ok=True)
        os.makedirs(os.path.join(evaluation_render_dir, 'ldr'), exist_ok=True)
        
        # render_train = True
        render_eval = True

        # for each time step
        for img_i in i_train:
           
            if img_i % 3 != 2:
                continue
            elif img_i == (num_img-1):
                continue
            
            count += 1
            # delta_t = args.step_frame // 2
            ratio = 0.5

            img_idx_embed_1 = img_i / num_img * 2. - 1.0
            img_idx_embed_2 = (img_i + 1) / num_img * 2. -1.0
            print(img_i)


            if img_i + 7 < (num_img-1):
                cam_i = img_i + 7
            else:
                cam_i = img_i - 6

            render_pose = poses[cam_i]
            R_w2t = render_pose[:3, :3].transpose(0,1)
            t_w2t = -torch.matmul(R_w2t, render_pose[:3, 3:4])
            
            ret1 = render_sm(img_idx_embed_1, 0, False,
                            num_img,
                            H, W, focal,
                            chunk=1024*16,
                            c2w=render_pose,
                            **render_kwargs_test)
            
            ret2 = render_sm(img_idx_embed_2, 0, False,
                                num_img,
                                H, W, focal,
                                chunk=1024*16,
                                c2w=render_pose,
                                **render_kwargs_test)
            
            T_i = torch.ones((1, H, W))
            final_rgb = torch.zeros((3, H, W))
            num_sample = ret1['raw_rgb'].shape[2]
            
            for j in range(0, num_sample):
                splat_alpha_dy_1, splat_rgb_dy_1, \
                splat_alpha_rig_1, splat_rgb_rig_1 = splat_rgb_img(ret1, ratio, R_w2t,
                                                                    t_w2t, j, H, W,
                                                                    focal, True)
                splat_alpha_dy_2, splat_rgb_dy_2, \
                splat_alpha_rig_2, splat_rgb_rig_2 = splat_rgb_img(ret2, 1. - ratio, R_w2t,
                                                                    t_w2t, j, H, W,
                                                                    focal, False)
                
                final_rgb += T_i * (splat_alpha_dy_1 * splat_rgb_dy_1 + \
                                    splat_alpha_rig_1 * splat_rgb_rig_1) * (1.0 - ratio)
                final_rgb += T_i * (splat_alpha_dy_2 * splat_rgb_dy_2 + \
                                    splat_alpha_rig_2 * splat_rgb_rig_2) * ratio
                
                alpha_1_final = (1.0 - (1. - splat_alpha_dy_1) * (1. - splat_alpha_rig_1)) * (1. - ratio)
                alpha_2_final = (1.0 - (1. - splat_alpha_dy_2) * (1. - splat_alpha_rig_2)) * ratio
                alpha_final = alpha_1_final + alpha_2_final
                
                T_i = T_i * (1.0 - alpha_final + 1e-10)               
            
            if Cam_param is not None:
                # tone_map = img_i // args.step_frame + 2
                tone_map = cam_i
                rgb8_ldr = to8b(Cam_param.RAD2LDR_img(final_rgb.permute(1,2,0), tone_map).cpu().numpy())
                
                mu = torch.tensor(50, dtype=final_rgb.dtype, device=final_rgb.device)
                final_rgb = torch.log(1 + mu * final_rgb) / torch.log(1 + mu)
                rgb8_hdr = to8b(final_rgb.permute(1,2,0).cpu().numpy())
                
                imageio.imwrite(os.path.join(evaluation_render_dir, 'hdr', '{:05d}.png'.format(img_i+count)), rgb8_hdr)
                imageio.imwrite(os.path.join(evaluation_render_dir, 'ldr', '{:05d}.png'.format(img_i+count)), rgb8_ldr)
            else:
                rgb8_ldr = to8b(final_rgb.permute(1,2,0).cpu().numpy())
                imageio.imwrite(os.path.join(evaluation_render_dir, 'ldr', '{:05d}.png'.format(img_i+count)), rgb8_ldr)

   
if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    evaluation()
