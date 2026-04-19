import configargparse

def config_parser():

    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, 
                        help='config file path')
    parser.add_argument("--expname", type=str, 
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/', 
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./data/llff/fern',

                        help='input data directory')
    parser.add_argument("--render_lockcam_slowmo", action='store_true', 
                        help='render fixed view + slowmo')
    parser.add_argument("--render_slowmo_bt", action='store_true', 
                        help='render space-time interpolation')

    parser.add_argument("--final_height", type=int, default=288, 
                        help='training image height, default is 512x288')
    
    # HDR-NSFF add-ons ##
    parser.add_argument("--sf_multires", type=int, default=0,
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--debug", action='store_true', help='debugging mode')
    parser.add_argument("--disp_model", type=str, default='midas', help='options: midas / depthcrafter / depth-anything')
    parser.add_argument("--sparse", action='store_true', help='sparse input or else')
    parser.add_argument("--half_res", action='store_true', default=True, help='option for blender dataset')

    parser.add_argument("--render_tm", type=str, default='tm', help='options: tm, wb, no')
    parser.add_argument("--semantic_flow", action='store_true', default=False, help='use semantic-guided optical flow (from DINO-tracker) instead of standard RAFT flow')
    parser.add_argument("--step_frame", type=int, default=1, help='step frame for loading data')
    parser.add_argument("--render_train", action='store_true', default=False, help='render train data')
    parser.add_argument("--share_crf", action='store_true', default=False, help='share crf')
    parser.add_argument("--share_wb", action='store_true', default=False, help='share just 3 wb')
    parser.add_argument("--log_scale", action='store_true', default=False, help='use log scale crf')
    parser.add_argument("--ref_idx", type=int, default=0, help='reference index for crf')
    parser.add_argument("--custom_times", type=int, default=None, help='render at just custom times')
    parser.add_argument("--leaky_th_h", type=float, default=0.9, help='leaky threshold high')
    parser.add_argument("--leaky_th_l", type=float, default=0.15, help='leaky threshold high')
    parser.add_argument("--viz_crf", action='store_true', default=False, help='optionsl Ture / False')
    parser.add_argument("--render_dy", action='store_true', default=False)
    parser.add_argument("--crf_start_step", type=int, default=0)
    parser.add_argument("--patch_size", type=int, default=32)
    parser.add_argument("--w_percept", type=float, default=0.01)
    parser.add_argument("--use_percept", action='store_true', default=False)
    parser.add_argument("--render_interpolate", action='store_true', default=False)
    parser.add_argument("--decay_crf_w", action='store_true', default=False)
    parser.add_argument("--crf_stop_step", type=int, default=9999999)
    parser.add_argument("--render_lockcam", action='store_true', default=False)
    parser.add_argument("--render_lockcam_slowmo_full", action='store_true', default=False)
    parser.add_argument("--stage", type=int, default=1)

    # difix options
    parser.add_argument("--use_difix", action='store_true', default=False)
    parser.add_argument("--sam2_repo", type=str, default="../nsff_scripts/sam2_repo",
                        help="Path to SAM2 repo. Defaults to ../nsff_scripts/sam2_repo relative to nsff_exp/")
    parser.add_argument("--sam2_checkpoint", type=str, default="../nsff_scripts/sam2_repo/checkpoints/sam2.1_hiera_large.pt",
                        help="Path to SAM2 checkpoint .pt file. Defaults to sam2_repo/checkpoints/sam2.1_hiera_large.pt")
    parser.add_argument("--sam2_model_cfg", type=str, default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--difix_interval", type=int, default=20000)
    parser.add_argument("--difix_num_views", type=int, default=2,
                        help='number of novel views synthesized per training frame')
    parser.add_argument("--difix_view_step", type=int, default=1,
                        help='camera index offset from training frame for novel view poses')
    parser.add_argument("--difix_start_step", type=int, default=200000)
    parser.add_argument("--use_render_l1", action='store_true', default=False)
    parser.add_argument("--difix_gt_prob", type=float, default=0.9,
                        help='probability of sampling a real GT view per iteration (vs difix-enhanced)')
    parser.add_argument("--w_render", type=float, default=1.)
    parser.add_argument("--difix_ref_scale", type=float, default=1.,
                        help='resize scale of reference GT image passed into Difix (reduce for VRAM)')
    parser.add_argument("--eval_start_time", type=int, default=0)
    parser.add_argument("--w_crf_smooth", type=float, default=0.001)
    parser.add_argument("--share_crf_lr_ratio", type=float, default=1.)

    # training options
    parser.add_argument("--netdepth", type=int, default=8, 
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256, 
                        help='channels per layer')
    parser.add_argument("--N_rand", type=int, default=32*32*4, 
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float, default=5e-4, 
                        help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=300, 
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--chunk", type=int, default=1024*128, 
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024*128, 
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_reload", action='store_true', 
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None, 
                        help='specific weights npy file to reload for coarse network')
    parser.add_argument("--N_iters", type=int, default=250001,
                        help='how many iteration do you want to run')

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64, 
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true', 
                        help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=0, 
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10, 
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4, 
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0., 
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')

    parser.add_argument("--render_bt", action='store_true', 
                        help='render bullet time')

    parser.add_argument("--render_test", action='store_true', 
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_factor", type=int, default=0, 
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='llff', 
                        help='options: llff / blender / deepvoxels')
    parser.add_argument("--testskip", type=int, default=8, 
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')
    ## blender flags
    parser.add_argument("--white_bkgd", action='store_true', 
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')

    ## llff flags
    parser.add_argument("--factor", type=int, default=8, 
                        help='downsample factor for LLFF images')
    parser.add_argument("--no_ndc", action='store_true', 
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--lindisp", action='store_true', 
                        help='sampling linearly in disparity rather than depth')
    parser.add_argument("--spherify", action='store_true', 
                        help='set for spherical 360 scenes')
    parser.add_argument("--target_idx", type=int, default=10, 
                        help='target_idx')
    parser.add_argument("--num_extra_sample", type=int, default=512, 
                        help='num_extra_sample')
    parser.add_argument("--decay_depth_w", action='store_true', 
                        help='decay depth weights')
    parser.add_argument("--use_motion_mask", action='store_true', 
                        help='use motion segmentation mask for hard-mining data-driven initialization')
    parser.add_argument("--decay_optical_flow_w", action='store_true', 
                        help='decay optical flow weights')

    parser.add_argument("--w_depth",   type=float, default=0.04, 
                        help='weights of depth loss')
    parser.add_argument("--w_optical_flow", type=float, default=0.02, 
                        help='weights of optical flow loss')
    parser.add_argument("--w_sm", type=float, default=0.1, 
                        help='weights of scene flow smoothness')
    parser.add_argument("--w_sf_reg", type=float, default=0.1, 
                        help='weights of scene flow regularization')
    parser.add_argument("--w_cycle", type=float, default=0.1, 
                        help='weights of cycle consistency')
    parser.add_argument("--w_prob_reg", type=float, default=0.1, 
                        help='weights of disocculusion weights')

    parser.add_argument("--w_entropy", type=float, default=1e-3, 
                        help='w_entropy regularization weight')

    parser.add_argument("--decay_iteration", type=int, default=50, 
                        help='data driven priors decay iteration * 1000')

    parser.add_argument("--chain_sf", action='store_true', 
                        help='5 frame consistency if true, \
                             otherwise 3 frame consistency')

    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--end_frame", type=int, default=50)

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=1000, 
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img",     type=int, default=10000, 
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=50000, 
                        help='frequency of weight ckpt saving')
    parser.add_argument("--use_tone_mapping", action='store_true', default=False, 
                        help='use tone mapping module')
    parser.add_argument("--use_initialize", action='store_true', default=False, 
                        help='use tone mapping module')
    parser.add_argument('--use_sat_mask', action='store_true', default=False)
    parser.add_argument('--tone_mapping', type=str, choices=["piece_wise", "nn", "hdr_hexplane"], default = "piece_wise")
    parser.add_argument('--render_mode', type=str, default='full',
                        help='full / cb / dynamic /rigid')
    return parser