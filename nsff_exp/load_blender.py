import os
import numpy as np
import imageio
import json
import cv2

def load_blender_data(basedir, start_frame, end_frame,
                      half_res=False, testskip=1,
                      factor=None, width=None, height=None,
                      load_imgs=True, evalution=False, 
                      target_idx=10, final_height=400, disp_model='midas'):
    print('factor ', factor)
    # splits = ['train', 'val', 'test']
    splits = ['train'] # First of all just consider training dataset
    metas =  {}
    for s in splits:
        with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
            metas[s] = json.load(fp)
            
    near = np.float32(metas['train']['near'])
    far = np.float32(metas['train']['far'])
    all_imgs = []
    all_poses = []
    counts = [0]
    
    for s in splits:
        meta = metas[s]
        imgs = []
        poses = []
        if s == 'train' or testskip == 0:
            skip = 1
        else:
            skip = testskip
            
        for frame in meta['frames'][::skip]:
            # fname = os.path.join(basedir, frame['file_path'] + '.png')
            fname = os.path.join(basedir, frame['file_path'])
            imgs.append(imageio.imread(fname))
            poses.append(np.array(frame['transform_matrix']))
            
        imgs = (np.array(imgs) / 255.).astype(np.float32) # keep all 4 channels (RGBA)
        poses = np.array(poses).astype(np.float32)
        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_poses.append(poses)
    
    i_split = [np.arange(counts[i], counts[i+1]) for i in range(len(splits))]

    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)
    
    imgs = imgs[start_frame:end_frame, ...]
    poses = poses[start_frame:end_frame, ...]
    
    H, W = imgs[0].shape[:2]
    camera_angle_x = float(meta['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)
    
    # import pdb; pdb.set_trace()
    
    if half_res:
        num_imgs = end_frame - start_frame
        resized_imgs = np.zeros((num_imgs, 400, 400, 4), dtype=imgs.dtype) # Change the number of images
        for i in range(imgs.shape[0]):
            resized_imgs[i] = cv2.resize(imgs[i], (400,400), interpolation=cv2.INTER_NEAREST)
        # imgs = cv2.resize(imgs, (400, 400), interpolation=cv2.INTER_NEAREST)
        # imgs = tf.image.resize_area(imgs, [400, 400]).numpy()
        imgs = resized_imgs
        H = H//2
        W = W//2
        focal = focal/2.
    
    # load disparity
    disp_dir = os.path.join(basedir, 'disp')
    dispfiles = [os.path.join(disp_dir, f) for f in sorted(os.listdir(disp_dir)) if f.endswith('npy')]
    dispfiles = dispfiles[start_frame:end_frame]
    disp = [cv2.resize(read_MiDaS_disp(f, 3.0),
                       (imgs.shape[2], imgs.shape[1]),
                       interpolation=cv2.INTER_NEAREST) for f in dispfiles]
    disp = np.stack(disp, -1) # [H,W,N]
    
    # load mask
    def imread(f):
        if f.endswith('png'):
            return imageio.imread(f, apply_gamma=False)
        else:
            return imageio.imread(f)
    
    mask_dir = os.path.join(basedir, 'motion_masks')
    maskfiles = [os.path.join(mask_dir, f) for f in sorted(os.listdir(mask_dir)) if f .endswith('png')]
    maskfiles = maskfiles[start_frame:end_frame]
    masks = [cv2.resize(imread(f)/255., (imgs.shape[2], imgs.shape[1]),
                        interpolation=cv2.INTER_NEAREST) for f in maskfiles]
    masks = np.stack(masks, -1)
    masks = np.float32(masks > 1e-3) # [H,W,N]
    
    # motion coordinates
    motion_coords = []
    for i in range(masks.shape[-1]):
        mask = masks[:,:,i]
        coord_y, coord_x = np.where(mask > 0.1)
        coord = np.stack((coord_y, coord_x), -1)
        motion_coords.append(coord)
    
    print(imgs.shape)
    print(disp.shape)
    
    c2w = poses[target_idx, :, :]
    
    # render_poses = render_wander_path(c2w)
    render_poses=c2w
    render_poses = np.array(render_poses).astype(np.float32)
    
    imgs = imgs.astype(np.float32)
    poses = poses.astype(np.float32)
    disp = np.moveaxis(disp, -1, 0).astype(np.float32)
    masks = np.moveaxis(masks, -1, 0).astype(np.float32)
    
    #disp, masks, poses, bds, render_poses, c2w, motio_coords 추가해서 output 해야함.
    
    return imgs, poses, render_poses, [H, W, focal], i_split, \
        disp, masks, c2w, motion_coords, near, far
    
    # render_poses = np.stack()

def read_MiDaS_disp(disp_fi, disp_rescale=10., h=None, w=None):
    disp = np.load(disp_fi)
    return disp
        
        
def render_wander_path(c2w):
    hwf = c2w[:,4:5]
    num_frames = 60
    max_disp = 48.0 # 64 , 48

    max_trans = max_disp / hwf[2][0] #self.targets['K_src'][0, 0, 0]  # Maximum camera translation to satisfy max_disp parameter
    output_poses = []

    for i in range(num_frames):
        x_trans = max_trans * np.sin(2.0 * np.pi * float(i) / float(num_frames))
        y_trans = max_trans * np.cos(2.0 * np.pi * float(i) / float(num_frames)) /3.0 #* 3.0 / 4.0
        z_trans = max_trans * np.cos(2.0 * np.pi * float(i) / float(num_frames)) /3.0

        i_pose = np.concatenate([
            np.concatenate(
                [np.eye(3), np.array([x_trans, y_trans, z_trans])[:, np.newaxis]], axis=1),
            np.array([0.0, 0.0, 0.0, 1.0])[np.newaxis, :]
        ],axis=0)#[np.newaxis, :, :]

        i_pose = np.linalg.inv(i_pose) #torch.tensor(np.linalg.inv(i_pose)).float()

        ref_pose = np.concatenate([c2w[:3, :4], np.array([0.0, 0.0, 0.0, 1.0])[np.newaxis, :]], axis=0)

        render_pose = np.dot(ref_pose, i_pose)
        # print('render_pose ', render_pose.shape)
        # sys.exit()
        output_poses.append(np.concatenate([render_pose[:3, :], hwf], 1))
    
    return output_poses