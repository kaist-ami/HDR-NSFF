import sys
sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image

from raft import RAFT
from core.utils import flow_viz
from core.utils.utils import InputPadder

from flow_utils import *
import skimage.morphology
import torchvision

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = 'cuda'
VIZ = True

def run_maskrcnn(model, img_path): #, intWidth=1024, intHeight=576):
    import PIL
    threshold = 0.5

    o_image = PIL.Image.open(img_path)
    
    # import pdb; pdb.set_trace()

    width, height = o_image.size

    if width > height:
        intWidth = 960
        intHeight = int(round( float(intWidth) / width * height))        
    else:
        intHeight = 960
        intWidth = int(round( float(intHeight) / height * width))        

    print('Semantic Seg Width %d Height %d'%(intWidth, intHeight))

    image = o_image.resize((intWidth, intHeight), PIL.Image.LANCZOS)

    image_tensor = torchvision.transforms.functional.to_tensor(image).cuda()

    if image_tensor.shape[0] == 4:
        image_tensor = image_tensor[:3,:,:]

    tenHumans = torch.FloatTensor(intHeight, intWidth).fill_(1.0).cuda()

    objPredictions = model([image_tensor])[0]

    for intMask in range(objPredictions['masks'].size(0)):
        if objPredictions['scores'][intMask].item() > threshold:
            if objPredictions['labels'][intMask].item() == 1: # person
                tenHumans[objPredictions['masks'][intMask, 0, :, :] > threshold] = 0.0

            if objPredictions['labels'][intMask].item() == 4: # motorcycle
                tenHumans[objPredictions['masks'][intMask, 0, :, :] > threshold] = 0.0

            if objPredictions['labels'][intMask].item() == 2: # bicycle
                tenHumans[objPredictions['masks'][intMask, 0, :, :] > threshold] = 0.0

            if objPredictions['labels'][intMask].item() == 8: # truck
                tenHumans[objPredictions['masks'][intMask, 0, :, :] > threshold] = 0.0

            if objPredictions['labels'][intMask].item() == 28: # umbrella
                tenHumans[objPredictions['masks'][intMask, 0, :, :] > threshold] = 0.0

            if objPredictions['labels'][intMask].item() == 17: # cat
                tenHumans[objPredictions['masks'][intMask, 0, :, :] > threshold] = 0.0

            if objPredictions['labels'][intMask].item() == 18: # dog
                tenHumans[objPredictions['masks'][intMask, 0, :, :] > threshold] = 0.0

            if objPredictions['labels'][intMask].item() == 36: # snowboard
                tenHumans[objPredictions['masks'][intMask, 0, :, :] > threshold] = 0.0

            if objPredictions['labels'][intMask].item() == 41: # skateboard
                tenHumans[objPredictions['masks'][intMask, 0, :, :] > threshold] = 0.0

    npyMask = skimage.morphology.erosion(tenHumans.cpu().numpy(),
                                         skimage.morphology.disk(1))
    npyMask = ((npyMask < 1e-3) * 255.0).clip(0.0, 255.0).astype(np.uint8)
    return npyMask


def motion_segmentation(basedir, threshold, multi_camera):
    import colmap_read_model as read_model

    points3dfile = os.path.join(basedir, 'sparse/points3D.bin')
    pts3d = read_model.read_points3d_binary(points3dfile)

    img_dir = glob.glob(basedir + '/images_*x*')[0]  
    # img0 = os.path.join(glob.glob(img_dir)[0], '%05d.png'%0)
    img0 = os.path.join(glob.glob(img_dir)[0], '%05d.png'%1)
    shape_0 = cv2.imread(img0).shape

    resized_height, resized_width = shape_0[0], shape_0[1]

    if multi_camera:
        imdata, perm, img_keys, HWF = load_colmap_data_multi_focal(basedir)
    else:
        imdata, perm, img_keys, hwf = load_colmap_data(basedir)
        scale_x, scale_y = resized_width / float(hwf[1]), resized_height / float(hwf[0])

        K = np.eye(3)
        K[0, 0] = hwf[2] * scale_x
        K[0, 2] = hwf[1] / 2. * scale_x
        K[1, 1] = hwf[2] * scale_y
        K[1, 2] = hwf[0] / 2. * scale_y

    xx = range(0, resized_width)
    yy = range(0, resized_height)  # , self.resized_h)
    xv, yv = np.meshgrid(xx, yy)
    p_ref = np.float32(np.stack((xv, yv), axis=-1))
    p_ref_h = np.reshape(p_ref, (-1, 2))
    p_ref_h = np.concatenate((p_ref_h, np.ones((p_ref_h.shape[0], 1))), axis=-1).T

    num_frames = len(perm) #- 1

    save_mask_dir = os.path.join(basedir, 'motion_segmentation')
    os.makedirs(save_mask_dir, exist_ok=True)

    # import pdb; pdb.set_trace()

    for i in range(0, num_frames): #len(perm) - 1):
        im_prev = imdata[img_keys[perm[max(0, i - 1)]]]
        im_ref = imdata[img_keys[perm[i]]]
        im_post = imdata[img_keys[perm[min(num_frames -1, i + 1)]]]

        if multi_camera:
            hwf = HWF[:, :, i]
            scale_x, scale_y = resized_width / float(hwf[1]), resized_height / float(hwf[0])

            K = np.eye(3)
            K[0, 0] = hwf[2] * scale_x
            K[0, 2] = hwf[1] / 2. * scale_x
            K[1, 1] = hwf[2] * scale_y
            K[1, 2] = hwf[0] / 2. * scale_y

            hwf_prev = HWF[:, :, max(0, i - 1)]
            hwf_ref = HWF[:, :, i]
            hwf_post = HWF[:, :, min(num_frames -1, i + 1)]

            K_prev = np.eye(3)
            K_prev[0, 0] = hwf_prev[2] * scale_x
            K_prev[0, 2] = hwf_prev[1] / 2. * scale_x
            K_prev[1, 1] = hwf_prev[2] * scale_y
            K_prev[1, 2] = hwf_prev[0] / 2. * scale_y

            K_post = np.eye(3)
            K_post[0, 0] = hwf_post[2] * scale_x
            K_post[0, 2] = hwf_post[1] / 2. * scale_x
            K_post[1, 1] = hwf_post[2] * scale_y
            K_post[1, 2] = hwf_post[0] / 2. * scale_y
        # import pdb; pdb.set_trace()

        # im_prev.name = im_prev.name.split('_')[1]
        # im_ref.name = im_ref.name.split('_')[1]
        # im_post.name = im_post.name.split('_')[1]

        print(im_prev.name, im_ref.name, im_post.name)

        T_prev_G = extract_poses(im_prev)        
        T_ref_G = extract_poses(im_ref)
        T_post_G = extract_poses(im_post)

        T_ref2prev = np.dot(T_prev_G, np.linalg.inv(T_ref_G))
        T_ref2post = np.dot(T_post_G, np.linalg.inv(T_ref_G))
        # load optical flow 
        if i == 0:
          fwd_flow, fwd_mask = read_optical_flow(basedir, 
                                       im_ref.name.split('_')[0], # shindy: if implement colmap gui, then use 0 instead 1
                                       read_fwd=True)
          bwd_flow = np.zeros_like(fwd_flow)
          bwd_mask = np.zeros_like(fwd_mask)

        elif i == num_frames - 1:
          bwd_flow, bwd_mask = read_optical_flow(basedir, 
                                       im_ref.name.split('_')[0], # shindy: if implement colmap gui, then use 0 instead 1
                                       read_fwd=False)
          fwd_flow = np.zeros_like(bwd_flow)
          fwd_mask = np.zeros_like(bwd_mask)

        else:
          fwd_flow, fwd_mask = read_optical_flow(basedir, 
                                       im_ref.name.split('_')[0], # shindy: if implement colmap gui, then use 0 instead 1
                                       read_fwd=True)
          bwd_flow, bwd_mask = read_optical_flow(basedir, 
                                       im_ref.name.split('_')[0], # shindy: if implement colmap gui, then use 0 instead 1
                                       read_fwd=False)

        p_post = p_ref + fwd_flow
        p_post_h = np.reshape(p_post, (-1, 2))
        p_post_h = np.concatenate((p_post_h, np.ones((p_post_h.shape[0], 1))), axis=-1).T

        if multi_camera: # Multi camera setting
            fwd_e_dist = compute_epipolar_distance_multi_focus(T_ref2post, K, K_post, 
                                               p_ref_h, p_post_h)
        else:   # Original
            fwd_e_dist = compute_epipolar_distance(T_ref2post, K, 
                                                   p_ref_h, p_post_h)  
        fwd_e_dist = np.reshape(fwd_e_dist, (fwd_flow.shape[0], fwd_flow.shape[1]))

        p_prev = p_ref + bwd_flow
        p_prev_h = np.reshape(p_prev, (-1, 2))
        p_prev_h = np.concatenate((p_prev_h, np.ones((p_prev_h.shape[0], 1))), axis=-1).T

        if multi_camera:
            bwd_e_dist = compute_epipolar_distance_multi_focus(T_ref2prev, K, K_prev, 
                                                p_ref_h, 
                                                p_prev_h)
        else:
            bwd_e_dist = compute_epipolar_distance(T_ref2prev, K, 
                                                p_ref_h, 
                                                p_prev_h)
        bwd_e_dist = np.reshape(bwd_e_dist, (bwd_flow.shape[0], bwd_flow.shape[1]))

        # import pdb; pdb.set_trace()

        # e_dist = np.maximum(bwd_e_dist, fwd_e_dist)
        # for non-video sequence
        e_dist = np.maximum(bwd_e_dist * bwd_mask, fwd_e_dist * fwd_mask)

        motion_mask = skimage.morphology.binary_opening(e_dist > threshold, skimage.morphology.disk(1))
        
        # import pdb; pdb.set_trace()


        # shindy: if implement colmap gui, then use 0 instead 1
        cv2.imwrite(os.path.join(save_mask_dir, im_ref.name.split('_')[0].replace('.jpg', '.png')), np.uint8(255 * (0. + motion_mask)))

    # RUN SEMANTIC SEGMENTATION
    img_dir = os.path.join(basedir, 'images')
    img_path_list = sorted(glob.glob(os.path.join(img_dir, '*.jpg'))) \
                    + sorted(glob.glob(os.path.join(img_dir, '*.png')))
    semantic_mask_dir = os.path.join(basedir, 'semantic_mask')
    netMaskrcnn = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True).cuda().eval()
    os.makedirs(semantic_mask_dir, exist_ok=True)

    for i in range(0, len(img_path_list)):
        img_path = img_path_list[i]
        img_name = img_path.split('/')[-1]
        semantic_mask = run_maskrcnn(netMaskrcnn, 
                                     img_path)
        cv2.imwrite(os.path.join(semantic_mask_dir, 
                                img_name.replace('.jpg', '.png')), 
                    semantic_mask)

    # combine them
    save_mask_dir = os.path.join(basedir, 'motion_masks')
    os.makedirs(save_mask_dir, exist_ok=True)

    mask_dir = os.path.join(basedir, 'motion_segmentation')
    mask_path_list = sorted(glob.glob(os.path.join(mask_dir, '*.png')))

    semantic_dir = os.path.join(basedir, 'semantic_mask')

    for mask_path in mask_path_list:
        print(mask_path)

        motion_mask = cv2.imread(mask_path)
        motion_mask = cv2.resize(motion_mask, (resized_width, resized_height), 
                                interpolation=cv2.INTER_NEAREST) 
        motion_mask = motion_mask[:, :, 0] > 0.1

        # combine from motion segmentation
        semantic_mask = cv2.imread(os.path.join(semantic_dir, mask_path.split('/')[-1]))
        semantic_mask = cv2.resize(semantic_mask, (resized_width, resized_height), 
                                interpolation=cv2.INTER_NEAREST)
        semantic_mask = semantic_mask[:, :, 0] > 0.1
        motion_mask = semantic_mask | motion_mask

        motion_mask = skimage.morphology.dilation(motion_mask, skimage.morphology.disk(2))
        cv2.imwrite(os.path.join(save_mask_dir, '%s'%mask_path.split('/')[-1]), 
                    np.uint8(np.clip((motion_mask), 0, 1) * 255) )
        # cv2.imwrite(os.path.join(mask_img_dir, '%s'%mask_path.split('/')[-1]), np.uint8(np.clip( (1. - motion_mask[..., np.newaxis]) * image, 0, 1) * 255) )

    # delete old mask dir
    os.system('rm -r %s'%mask_dir)
    os.system('rm -r %s'%semantic_dir)


def motion_segmentation_blender(basedir, threshold, multi_camera, i_step):

    from blender_read_model import BlenderDataset

    img_dir = glob.glob(basedir + '/images_*x*')[0]  
    # img0 = os.path.join(glob.glob(img_dir)[0], '%05d.png'%0)
    img0 = os.path.join(glob.glob(img_dir)[0], '%05d.png'%1)
    shape_0 = cv2.imread(img0).shape

    resized_height, resized_width = shape_0[0], shape_0[1]

    blender_dir = os.path.join(basedir, '..')

    train_dataset = BlenderDataset(
        blender_dir,
        "train",
        1.0
    )

    w, h = train_dataset.img_wh
    f = train_dataset.focal
    hwf = np.array([h, w, f]).reshape([3,1])

    if multi_camera:
        pass
        # imdata, perm, img_keys, HWF = load_colmap_data_multi_focal(basedir)
    else:
        # imdata, perm, img_keys, hwf = load_colmap_data(basedir)
        scale_x, scale_y = resized_width / float(hwf[1]), resized_height / float(hwf[0])

        K = np.eye(3)
        K[0, 0] = hwf[2] * scale_x
        K[0, 2] = hwf[1] / 2. * scale_x
        K[1, 1] = hwf[2] * scale_y
        K[1, 2] = hwf[0] / 2. * scale_y

    xx = range(0, resized_width)
    yy = range(0, resized_height)  # , self.resized_h)
    xv, yv = np.meshgrid(xx, yy)
    p_ref = np.float32(np.stack((xv, yv), axis=-1))
    p_ref_h = np.reshape(p_ref, (-1, 2))
    p_ref_h = np.concatenate((p_ref_h, np.ones((p_ref_h.shape[0], 1))), axis=-1).T

    # num_frames = len(perm) #- 1

    num_frames = len(train_dataset.poses)

    save_mask_dir = os.path.join(basedir, 'motion_segmentation')
    os.makedirs(save_mask_dir, exist_ok=True)
    
    # import pdb; pdb.set_trace()

    for i in range(47): #len(perm) - 1): 47 for 5step in lego_room

        im_prev = train_dataset.image_paths[max(0, i-1)]
        im_ref = train_dataset.image_paths[i]
        im_post = train_dataset.image_paths[min(num_frames - 1, i + 1)]

        if multi_camera:
            pass
            # hwf = HWF[:, :, i]
            # scale_x, scale_y = resized_width / float(hwf[1]), resized_height / float(hwf[0])

            # K = np.eye(3)
            # K[0, 0] = hwf[2] * scale_x
            # K[0, 2] = hwf[1] / 2. * scale_x
            # K[1, 1] = hwf[2] * scale_y
            # K[1, 2] = hwf[0] / 2. * scale_y

            # hwf_prev = HWF[:, :, max(0, i - 1)]
            # hwf_ref = HWF[:, :, i]
            # hwf_post = HWF[:, :, min(num_frames -1, i + 1)]

            # K_prev = np.eye(3)
            # K_prev[0, 0] = hwf_prev[2] * scale_x
            # K_prev[0, 2] = hwf_prev[1] / 2. * scale_x
            # K_prev[1, 1] = hwf_prev[2] * scale_y
            # K_prev[1, 2] = hwf_prev[0] / 2. * scale_y

            # K_post = np.eye(3)
            # K_post[0, 0] = hwf_post[2] * scale_x
            # K_post[0, 2] = hwf_post[1] / 2. * scale_x
            # K_post[1, 1] = hwf_post[2] * scale_y
            # K_post[1, 2] = hwf_post[0] / 2. * scale_y
        # import pdb; pdb.set_trace()

        # im_prev.name = im_prev.name.split('_')[1]
        # im_ref.name = im_ref.name.split('_')[1]
        # im_post.name = im_post.name.split('_')[1]

        print(im_prev, im_ref, im_post)

        # np.linalg.inv(w2c_mats)

        # import pdb; pdb.set_trace()

        # T_prev_G = train_dataset.poses[max(0, i-1)]
        # T_ref_G = train_dataset.poses[i]
        # T_post_G = train_dataset.poses[min(num_frames - 1, i + 1)]

        # # import pdb; pdb.set_trace()

        T_prev_G = np.linalg.inv(train_dataset.poses[max(0, i-1)]) # c2w to w2c by inverse
        T_ref_G = np.linalg.inv(train_dataset.poses[i])
        T_post_G = np.linalg.inv(train_dataset.poses[min(num_frames - 1, i + 1)])
        
        T_prev_G = np.concatenate([T_prev_G[1,:], T_prev_G[0,:], -T_prev_G[2,:], T_prev_G[3,:]]).reshape([4,4])
        T_ref_G = np.concatenate([T_ref_G[1,:], T_ref_G[0,:], -T_ref_G[2,:], T_ref_G[3,:]]).reshape([4,4])
        T_post_G = np.concatenate([T_post_G[1,:], T_post_G[0,:], -T_post_G[2,:], T_post_G[3,:]]).reshape([4,4])
        
        # T_prev_G = train_dataset.poses[max(0, i-1)]
        # T_ref_G = train_dataset.poses[i]
        # T_post_G = train_dataset.poses[min(num_frames - 1, i + 1)]
        
        # mat = np.stack([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]], axis=0)
        
        # T_prev_G = T_prev_G @ mat
        # T_ref_G = T_ref_G @ mat
        # T_post_G = T_post_G @ mat

        img_dir = os.path.join(basedir, 'images')
        img_path_list = sorted(glob.glob(os.path.join(img_dir, '*.jpg'))) \
                    + sorted(glob.glob(os.path.join(img_dir, '*.png')))
        
        im_ref_name = img_path_list[i]

        # T_prev_G = extract_poses(im_prev)        
        # T_ref_G = extract_poses(im_ref)
        # T_post_G = extract_poses(im_post)

        T_ref2prev = np.dot(T_prev_G, np.linalg.inv(T_ref_G))
        T_ref2post = np.dot(T_post_G, np.linalg.inv(T_ref_G))
        # load optical flow 
        if i == 0:
          fwd_flow, fwd_mask = read_optical_flow(basedir, 
                                       im_ref_name.split('/')[-1],
                                       read_fwd=True)
          bwd_flow = np.zeros_like(fwd_flow)
          bwd_mask = np.zeros_like(fwd_mask)

        elif i == num_frames - 1:
          bwd_flow, bwd_mask = read_optical_flow(basedir, 
                                       im_ref_name.split('/')[-1],
                                       read_fwd=False)
          fwd_flow = np.zeros_like(bwd_flow)
          fwd_mask = np.zeros_like(bwd_mask)

        else:
          fwd_flow, fwd_mask = read_optical_flow(basedir, 
                                       im_ref_name.split('/')[-1],
                                       read_fwd=True)
          bwd_flow, bwd_mask = read_optical_flow(basedir, 
                                       im_ref_name.split('/')[-1],
                                       read_fwd=False)

        p_post = p_ref + fwd_flow
        p_post_h = np.reshape(p_post, (-1, 2))
        p_post_h = np.concatenate((p_post_h, np.ones((p_post_h.shape[0], 1))), axis=-1).T

        if multi_camera: # Multi camera setting
            pass
            # fwd_e_dist = compute_epipolar_distance_multi_focus(T_ref2post, K, K_post, 
            #                                    p_ref_h, p_post_h)
        else:   # Original
            fwd_e_dist = compute_epipolar_distance(T_ref2post, K, 
                                                   p_ref_h, p_post_h)  
        fwd_e_dist = np.reshape(fwd_e_dist, (fwd_flow.shape[0], fwd_flow.shape[1]))

        p_prev = p_ref + bwd_flow
        p_prev_h = np.reshape(p_prev, (-1, 2))
        p_prev_h = np.concatenate((p_prev_h, np.ones((p_prev_h.shape[0], 1))), axis=-1).T

        if multi_camera:
            pass
            # bwd_e_dist = compute_epipolar_distance_multi_focus(T_ref2prev, K, K_prev, 
            #                                     p_ref_h, 
            #                                     p_prev_h)
        else:
            bwd_e_dist = compute_epipolar_distance(T_ref2prev, K, 
                                                p_ref_h, 
                                                p_prev_h)
        bwd_e_dist = np.reshape(bwd_e_dist, (bwd_flow.shape[0], bwd_flow.shape[1]))

        # import pdb; pdb.set_trace()

        # e_dist = np.maximum(bwd_e_dist, fwd_e_dist)
        # for non-video sequence
        e_dist = np.maximum(bwd_e_dist * bwd_mask, fwd_e_dist * fwd_mask)

        motion_mask = skimage.morphology.binary_opening(e_dist > threshold, skimage.morphology.disk(1))


        # shindy: if implement colmap gui, then use 0 instead 1
        cv2.imwrite(os.path.join(save_mask_dir, im_ref_name.split('/')[-1].replace('.jpg', '.png')), np.uint8(255 * (0. + motion_mask)))
        
        
        # For checking epipolar line (L485 ~ L498)
        img_dir = os.path.join(basedir, 'images_*')
        img_path_list = sorted(glob.glob(os.path.join(img_dir, '*.jpg'))) \
                    + sorted(glob.glob(os.path.join(img_dir, '*.png')))
        
        # p_1 = np.array([320, 240, 1])
        p_1 = np.array([p_ref_h[:, 25000], p_ref_h[:, 45000], p_ref_h[:, 65000], p_ref_h[:, 85000], p_ref_h[:, 105000], p_ref_h[:, 125000], p_ref_h[:, 144900], p_ref_h[:, 145000]])
        p_2 = np.array([p_post_h[:, 25000], p_post_h[:, 45000], p_post_h[:, 65000], p_post_h[:, 85000], p_post_h[:, 105000], p_post_h[:, 125000], p_post_h[:,144900], p_post_h[:, 145000]])
        
        img1 = cv2.imread(img_path_list[i])
        img2 = cv2.imread(img_path_list[min(48, i+1)])
        
        visualize_epiploar_on_images(T_ref2post, K, img1, img2, p_1, p_2, i)
        

    # RUN SEMANTIC SEGMENTATION
    img_dir = os.path.join(basedir, 'images')
    img_path_list = sorted(glob.glob(os.path.join(img_dir, '*.jpg'))) \
                    + sorted(glob.glob(os.path.join(img_dir, '*.png')))
    semantic_mask_dir = os.path.join(basedir, 'semantic_mask')
    netMaskrcnn = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True).cuda().eval()
    os.makedirs(semantic_mask_dir, exist_ok=True)

    for i in range(0, len(img_path_list)):
        img_path = img_path_list[i]
        img_name = img_path.split('/')[-1]
        semantic_mask = run_maskrcnn(netMaskrcnn, 
                                     img_path)
        cv2.imwrite(os.path.join(semantic_mask_dir, 
                                img_name.replace('.jpg', '.png')), 
                    semantic_mask)

    # combine them
    save_mask_dir = os.path.join(basedir, 'motion_masks')
    os.makedirs(save_mask_dir, exist_ok=True)

    mask_dir = os.path.join(basedir, 'motion_segmentation')
    mask_path_list = sorted(glob.glob(os.path.join(mask_dir, '*.png')))

    semantic_dir = os.path.join(basedir, 'semantic_mask')

    for mask_path in mask_path_list:
        print(mask_path)

        motion_mask = cv2.imread(mask_path)
        motion_mask = cv2.resize(motion_mask, (resized_width, resized_height), 
                                interpolation=cv2.INTER_NEAREST) 
        motion_mask = motion_mask[:, :, 0] > 0.1

        # combine from motion segmentation
        semantic_mask = cv2.imread(os.path.join(semantic_dir, mask_path.split('/')[-1]))
        semantic_mask = cv2.resize(semantic_mask, (resized_width, resized_height), 
                                interpolation=cv2.INTER_NEAREST)
        semantic_mask = semantic_mask[:, :, 0] > 0.1
        motion_mask = semantic_mask | motion_mask

        motion_mask = skimage.morphology.dilation(motion_mask, skimage.morphology.disk(2))
        cv2.imwrite(os.path.join(save_mask_dir, '%s'%mask_path.split('/')[-1]), 
                    np.uint8(np.clip((motion_mask), 0, 1) * 255) )
        # cv2.imwrite(os.path.join(mask_img_dir, '%s'%mask_path.split('/')[-1]), np.uint8(np.clip( (1. - motion_mask[..., np.newaxis]) * image, 0, 1) * 255) )

    # delete old mask dir
    os.system('rm -r %s'%mask_dir)
    os.system('rm -r %s'%semantic_dir)


def load_image(imfile):
    long_dim = 768 
    # long_dim = 854
    # long_dim=509 # shindy modified 768 to 509 for dataset jun_running

    img = np.array(Image.open(imfile)).astype(np.uint8)

    # Portrait Orientation
    if img.shape[0] > img.shape[1]:
        input_h = long_dim
        input_w = int(round( float(input_h) / img.shape[0] * img.shape[1]))
    # Landscape Orientation
    else:
        input_w = long_dim 
        input_h = int(round( float(input_w) / img.shape[1] * img.shape[0]))

    print('flow input w %d h %d'%(input_w, input_h))
    img = cv2.resize(img, (input_w, input_h), cv2.INTER_LINEAR)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img


def load_image_list(image_files):
    images = []
    for imfile in sorted(image_files):
        images.append(load_image(imfile))

    images = torch.stack(images, dim=0)
    images = images.to(DEVICE)

    padder = InputPadder(images.shape)
    return padder.pad(images)[0]

def load_image_list_step(image_files, step, offset=0):
    images = []
    for imfile in sorted(image_files)[offset::step]:
        images.append(load_image(imfile))
    
    images = torch.stack(images, dim=0)
    images = images.to(DEVICE)
    
    padder = InputPadder(images.shape)
    return padder.pad(images)[0]

def viz(img, flo):
    img = img[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1,2,0).cpu().numpy()
    
    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img, flo], axis=0)

    cv2.imshow('image', img_flo[:, :, [2,1,0]]/255.0)
    cv2.waitKey()


def resize_flow(flow, img_h, img_w):
    # flow = np.load(flow_path)

    flow = flow.astype(np.float32)
    flow_h, flow_w = flow.shape[0], flow.shape[1]
    flow[:, :, 0] *= float(img_w)/float(flow_w)
    flow[:, :, 1] *= float(img_h)/float(flow_h)
    flow = cv2.resize(flow, (img_w, img_h), cv2.INTER_LINEAR)

    return flow

def read_img(img_dir, img1_name, img2_name):
    return cv2.imread(os.path.join(img_dir, img1_name + '.png')), \
        cv2.imread(os.path.join(img_dir, img2_name + '.png'))

def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow_new = flow.copy()
    flow_new[:,:,0] += np.arange(w)
    flow_new[:,:,1] += np.arange(h)[:,np.newaxis]

    res = cv2.remap(img, flow_new, None, cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
    return res

def refinement_flow(fwd_flow, img1, img2):
    flow_refine = cv2.VariationalRefinement.create()

    refine_flow = flow_refine.calc(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), 
                                 cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY), 
                                 fwd_flow)

    return refine_flow

def compute_fwdbwd_mask(fwd_flow, bwd_flow):
    alpha_1 = 0.5
    alpha_2 = 0.5

    bwd2fwd_flow = warp_flow(bwd_flow, fwd_flow)
    fwd_lr_error = np.linalg.norm(fwd_flow + bwd2fwd_flow, axis=-1)
    fwd_mask = fwd_lr_error < alpha_1  * (np.linalg.norm(fwd_flow, axis=-1) \
                + np.linalg.norm(bwd2fwd_flow, axis=-1)) + alpha_2

    fwd2bwd_flow = warp_flow(fwd_flow, bwd_flow)
    bwd_lr_error = np.linalg.norm(bwd_flow + fwd2bwd_flow, axis=-1)

    bwd_mask = bwd_lr_error < alpha_1  * (np.linalg.norm(bwd_flow, axis=-1) \
                + np.linalg.norm(fwd2bwd_flow, axis=-1)) + alpha_2

    return fwd_mask, bwd_mask

def run_optical_flows(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    basedir = "%s"%args.data_path
    # print(basedir)
    img_dir = glob.glob(basedir + '/images_*')[0]  #basedir + '/images_*288' 
    # img_path_train = os.path.join(glob.glob(img_dir)[0], '%05d.png'%0)
    img_path_train = os.path.join(glob.glob(img_dir)[0], '%05d.png'%1)
    img_train = cv2.imread(img_path_train)

    # interval = 1
    interval = args.step
    of_dir = os.path.join(basedir, 'flow_i%d'%interval)

    if not os.path.exists(of_dir):
        os.makedirs(of_dir)

    with torch.no_grad():
        images = glob.glob(os.path.join(basedir, 'images/', '*.png')) + \
                 glob.glob(os.path.join(basedir, 'images/', '*.jpg'))
                 
        if args.step != 1: # Using step
            images = load_image_list_step(images, args.step)
        else:
            images = load_image_list(images)
            
        if args.traj_init:
            trajs = sorted(glob.glob(os.path.join(basedir, 'semantic_flow/', '*.npz')))
            trajs = [np.load(traj) for traj in trajs]
            trajs = [torch.from_numpy(traj['flow']).permute(2, 0, 1).float().to(DEVICE) for traj in trajs]  
            trajs = torch.stack(trajs, dim=0)
            trajs = trajs.to(DEVICE)
            
            
        for i in range(images.shape[0]-1):
            print(i)
            image1 = images[i,None]
            image2 = images[i + 1,None]
            
            if args.traj_init:
                traj_idx = i * 2
                traj_fwd_o = trajs[traj_idx, None]
                traj_bwd_o = trajs[traj_idx + 1, None]
                traj_fwd = F.interpolate(traj_fwd_o, (image1.shape[2] // 8, image1.shape[3] // 8), mode='bilinear', align_corners=False) // 8
                traj_bwd = F.interpolate(traj_bwd_o, (image1.shape[2] // 8, image1.shape[3] // 8), mode='bilinear', align_corners=False) // 8
                
            else:
                traj_fwd = None
                traj_bwd = None

            if image1.size(1) == 4:
                image1 = image1[:,0:3,:,:]
                image2 = image2[:,0:3,:,:]

            _, flow_up_fwd = model(image1, image2, iters=20, test_mode=True, flow_init=traj_fwd)
            _, flow_up_bwd = model(image2, image1, iters=20, test_mode=True, flow_init=traj_bwd)

            flow_up_fwd = flow_up_fwd[0].cpu().numpy().transpose(1, 2, 0)
            flow_up_bwd = flow_up_bwd[0].cpu().numpy().transpose(1, 2, 0)

            img1 = cv2.resize(np.uint8(np.clip(image1[0].cpu().numpy().transpose(1, 2, 0), 0, 255)), 
                             (img_train.shape[1], img_train.shape[0]), 
                              cv2.INTER_LINEAR)
            img2 = cv2.resize(np.uint8(np.clip(image2[0].cpu().numpy().transpose(1, 2, 0), 0, 255)), 
                             (img_train.shape[1], img_train.shape[0]), 
                             cv2.INTER_LINEAR)

            fwd_flow = resize_flow(flow_up_fwd, 
                                   img_train.shape[0], 
                                   img_train.shape[1])
            # fwd_flow = refinement_flow(fwd_flow, img1, img2)

            bwd_flow = resize_flow(flow_up_bwd, 
                                   img_train.shape[0], 
                                   img_train.shape[1])
            # bwd_flow = refinement_flow(bwd_flow, img1, img2)

            fwd_mask, bwd_mask = compute_fwdbwd_mask(fwd_flow, 
                                                     bwd_flow)

            if VIZ:
                viz_flow_dir = basedir + "/viz_flow_i%d"%interval
                if not os.path.exists(viz_flow_dir):
                    os.makedirs(viz_flow_dir)

                viz_warp_dir = basedir + "/viz_warp_imgs_i%d"%interval
                if not os.path.exists(viz_warp_dir):
                    os.makedirs(viz_warp_dir)
                    
                if args.traj_init:
                    # import pdb; pdb.set_trace()
                    viz_traj_dir = basedir + "/viz_traj"
                    os.makedirs(viz_traj_dir, exist_ok=True)
                    fwd_traj_o = traj_fwd_o.squeeze(0).permute(1,2,0).cpu().numpy()
                    bwd_traj_o = traj_bwd_o.squeeze(0).permute(1,2,0).cpu().numpy()
                    
                    fwd_traj = resize_flow(traj_fwd.squeeze(0).permute(1,2,0).cpu().numpy(), img_train.shape[0], img_train.shape[1])
                    bwd_traj = resize_flow(traj_bwd.squeeze(0).permute(1,2,0).cpu().numpy(), img_train.shape[0], img_train.shape[1])
                    
                    plt.figure(figsize=(12, 6))
                    plt.subplot(2,3,1)
                    plt.imshow(img1)
                    plt.subplot(2,3,4)
                    plt.imshow(img2)
                    
                    plt.subplot(2,3,2)
                    plt.imshow(flow_viz.flow_to_image(fwd_traj)/255.)
                    plt.subplot(2,3,5)
                    plt.imshow(flow_viz.flow_to_image(bwd_traj)/255.)
                    
                    plt.subplot(2,3,3)
                    plt.imshow(flow_viz.flow_to_image(fwd_traj_o)/255.)
                    plt.subplot(2,3,6)
                    plt.imshow(flow_viz.flow_to_image(bwd_traj_o)/255.)
                    
                    plt.savefig(viz_traj_dir + '/%02d.jpg'%i)
                    plt.close()
                    

                plt.figure(figsize=(12, 6))
                plt.subplot(2,3,1)
                plt.imshow(img1)
                plt.subplot(2,3,4)
                plt.imshow(img2)

                plt.subplot(2,3,2)
                plt.imshow(flow_viz.flow_to_image(fwd_flow)/255.)
                plt.subplot(2,3,3)
                plt.imshow(flow_viz.flow_to_image(bwd_flow)/255.)

                plt.subplot(2,3,5)
                plt.imshow(flow_viz.flow_to_image(fwd_flow)/255. * np.float32(fwd_mask[..., np.newaxis]))

                plt.subplot(2,3,6)
                plt.imshow(flow_viz.flow_to_image(bwd_flow)/255. * np.float32(bwd_mask[..., np.newaxis]))

                # plt.savefig('./viz_flow/%02d.jpg'%i)
                plt.savefig(viz_flow_dir + '/%02d.jpg'%i)
                plt.close()

                warped_im2 = warp_flow(img2, fwd_flow)
                warped_im0 = warp_flow(img1, bwd_flow)

                viz_flow_dir_ind = basedir + "/viz_individiaul_flow_i%d"%interval
                os.makedirs(viz_flow_dir_ind, exist_ok=True)   

                cv2.imwrite(viz_flow_dir_ind + '/img_%05d_fwd.jpg'%(i), flow_viz.flow_to_image(fwd_flow)[..., ::-1])
                cv2.imwrite(viz_flow_dir_ind + '/img_%05d_bwd.jpg'%(i+1), flow_viz.flow_to_image(bwd_flow)[..., ::-1])
  
                # cv2.imwrite('./viz_warp_imgs/im_%05d.jpg'%(i), img1[..., ::-1])
                # cv2.imwrite('./viz_warp_imgs/im_%05d_fwd.jpg'%(i), warped_im2[..., ::-1])
                # cv2.imwrite('./viz_warp_imgs/im_%05d_bwd.jpg'%(i + 1), warped_im0[..., ::-1])

                cv2.imwrite(viz_warp_dir + '/im_%05d.jpg'%(i), img1[..., ::-1])
                cv2.imwrite(viz_warp_dir + '/im_%05d_fwd.jpg'%(i), warped_im2[..., ::-1])
                cv2.imwrite(viz_warp_dir + '/im_%05d_bwd.jpg'%(i + 1), warped_im0[..., ::-1])

            
            # if image start with 00001.png, use i+1, i+2 instead start with 00000.png then use i, i+1
            # np.savez(os.path.join(of_dir, '%05d_fwd.npz'%(i + 1)), flow=fwd_flow, mask=fwd_mask)
            # np.savez(os.path.join(of_dir, '%05d_bwd.npz'%(i + 2)), flow=bwd_flow, mask=bwd_mask)

            np.savez(os.path.join(of_dir, '%05d_fwd.npz'%(i)), flow=fwd_flow, mask=fwd_mask)
            np.savez(os.path.join(of_dir, '%05d_bwd.npz'%(i + 1)), flow=bwd_flow, mask=bwd_mask)

def run_optical_flows_mask(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    basedir = "%s"%args.data_path
    # print(basedir)
    img_dir = glob.glob(basedir + '/images_*')[0]  #basedir + '/images_*288' 

    # import pdb; pdb.set_trace() 

    # img_path_train = os.path.join(glob.glob(img_dir)[0], '%05d.png'%0)
    img_path_train = os.path.join(glob.glob(img_dir)[0], '%05d.png'%1)
    img_train = cv2.imread(img_path_train)

    # interval = 1
    interval = args.step
    of_dir = os.path.join(basedir, 'flow_i%d'%interval)

    if not os.path.exists(of_dir):
        os.makedirs(of_dir)

    with torch.no_grad():
        images = glob.glob(os.path.join(basedir, 'images/', '*.png')) + \
                 glob.glob(os.path.join(basedir, 'images/', '*.jpg'))
                 
        if args.step != 1: # Using step
            images = load_image_list_step(images, args.step)
        else:
            images = load_image_list(images)
            
        if args.traj_init:
            trajs = sorted(glob.glob(os.path.join(basedir, 'semantic_flow/', '*.npz')))
            trajs = [np.load(traj) for traj in trajs]
            trajs = [torch.from_numpy(traj['flow']).permute(2, 0, 1).float().to(DEVICE) for traj in trajs]  
            trajs = torch.stack(trajs, dim=0)
            trajs = trajs.to(DEVICE)

        # import pdb; pdb.set_trace()
        masks = sorted(glob.glob(os.path.join(basedir, 'motion_masks', '*g')))
        # masks = load_image_list_step(masks, args.step)


        for i in range(images.shape[0]-1):
            print(i)
            image1 = images[i,None]
            image2 = images[i + 1,None]
            
            if args.traj_init:
                traj_idx = i * 2
                traj_fwd_o = trajs[traj_idx, None]
                traj_bwd_o = trajs[traj_idx + 1, None]
                traj_fwd = F.interpolate(traj_fwd_o, (image1.shape[2] // 8, image1.shape[3] // 8), mode='bilinear', align_corners=False) // 8
                traj_bwd = F.interpolate(traj_bwd_o, (image1.shape[2] // 8, image1.shape[3] // 8), mode='bilinear', align_corners=False) // 8
                
            else:
                traj_fwd = None
                traj_bwd = None

            # import pdb; pdb.set_trace()

            if image1.size(1) == 4:
                image1 = image1[:,0:3,:,:]
                image2 = image2[:,0:3,:,:]
            # print(image1.shape)

            _, flow_up_fwd = model(image1, image2, iters=20, test_mode=True, flow_init=traj_fwd)
            _, flow_up_bwd = model(image2, image1, iters=20, test_mode=True, flow_init=traj_bwd)

            flow_up_fwd = flow_up_fwd[0].cpu().numpy().transpose(1, 2, 0)
            flow_up_bwd = flow_up_bwd[0].cpu().numpy().transpose(1, 2, 0)

            img1 = cv2.resize(np.uint8(np.clip(image1[0].cpu().numpy().transpose(1, 2, 0), 0, 255)), 
                             (img_train.shape[1], img_train.shape[0]), 
                              cv2.INTER_LINEAR)
            img2 = cv2.resize(np.uint8(np.clip(image2[0].cpu().numpy().transpose(1, 2, 0), 0, 255)), 
                             (img_train.shape[1], img_train.shape[0]), 
                             cv2.INTER_LINEAR)

            fwd_flow = resize_flow(flow_up_fwd, 
                                   img_train.shape[0], 
                                   img_train.shape[1])
            # fwd_flow = refinement_flow(fwd_flow, img1, img2)

            bwd_flow = resize_flow(flow_up_bwd, 
                                   img_train.shape[0], 
                                   img_train.shape[1])
            # bwd_flow = refinement_flow(bwd_flow, img1, img2)

            fwd_mask, bwd_mask = compute_fwdbwd_mask(fwd_flow, 
                                                     bwd_flow)
            
            # import pdb; pdb.set_trace()
            mask_1_path = masks[i]
            mask_1 = cv2.imread(mask_1_path, cv2.IMREAD_GRAYSCALE)
            mask_1 = cv2.resize(mask_1, (fwd_mask.shape[1], fwd_mask.shape[0]), interpolation=cv2.INTER_NEAREST)

            fwd_mask = fwd_mask * (mask_1 > 0)

            mask_2_path = masks[i+1]
            mask_2 = cv2.imread(mask_2_path, cv2.IMREAD_GRAYSCALE)
            mask_2 = cv2.resize(mask_2, (fwd_mask.shape[1], fwd_mask.shape[0]), interpolation=cv2.INTER_NEAREST)

            fwd_mask = fwd_mask * (mask_1 > 0)
            bwd_mask = bwd_mask * (mask_2 > 0)


            if VIZ:
                viz_flow_dir = basedir + "/viz_flow_i%d"%interval
                if not os.path.exists(viz_flow_dir):
                    os.makedirs(viz_flow_dir)

                viz_warp_dir = basedir + "/viz_warp_imgs_i%d"%interval
                if not os.path.exists(viz_warp_dir):
                    os.makedirs(viz_warp_dir)

                viz_flow_dir_ind = basedir + "/viz_individiaul_flow_i%d"%interval
                os.makedirs(viz_flow_dir_ind, exist_ok=True)
                    
                if args.traj_init:
                    # import pdb; pdb.set_trace()
                    viz_traj_dir = basedir + "/viz_traj"
                    os.makedirs(viz_traj_dir, exist_ok=True)
                    fwd_traj_o = traj_fwd_o.squeeze(0).permute(1,2,0).cpu().numpy()
                    bwd_traj_o = traj_bwd_o.squeeze(0).permute(1,2,0).cpu().numpy()
                    
                    fwd_traj = resize_flow(traj_fwd.squeeze(0).permute(1,2,0).cpu().numpy(), img_train.shape[0], img_train.shape[1])
                    bwd_traj = resize_flow(traj_bwd.squeeze(0).permute(1,2,0).cpu().numpy(), img_train.shape[0], img_train.shape[1])
                    
                    plt.figure(figsize=(12, 6))
                    plt.subplot(2,3,1)
                    plt.imshow(img1)
                    plt.subplot(2,3,4)
                    plt.imshow(img2)
                    
                    plt.subplot(2,3,2)
                    plt.imshow(flow_viz.flow_to_image(fwd_traj)/255.)
                    plt.subplot(2,3,5)
                    plt.imshow(flow_viz.flow_to_image(bwd_traj)/255.)
                    
                    plt.subplot(2,3,3)
                    plt.imshow(flow_viz.flow_to_image(fwd_traj_o)/255.)
                    plt.subplot(2,3,6)
                    plt.imshow(flow_viz.flow_to_image(bwd_traj_o)/255.)
                    
                    plt.savefig(viz_traj_dir + '/%02d.jpg'%i)
                    plt.close()
                    

                plt.figure(figsize=(12, 6))
                plt.subplot(2,3,1)
                plt.imshow(img1)
                plt.subplot(2,3,4)
                plt.imshow(img2)

                plt.subplot(2,3,2)
                plt.imshow(flow_viz.flow_to_image(fwd_flow)/255.)
                plt.subplot(2,3,3)
                plt.imshow(flow_viz.flow_to_image(bwd_flow)/255.)

                plt.subplot(2,3,5)
                plt.imshow(flow_viz.flow_to_image(fwd_flow)/255. * np.float32(fwd_mask[..., np.newaxis]))

                plt.subplot(2,3,6)
                plt.imshow(flow_viz.flow_to_image(bwd_flow)/255. * np.float32(bwd_mask[..., np.newaxis]))

                # plt.savefig('./viz_flow/%02d.jpg'%i)
                plt.savefig(viz_flow_dir + '/%02d.jpg'%i)
                plt.close()

                warped_im2 = warp_flow(img2, fwd_flow)
                warped_im0 = warp_flow(img1, bwd_flow)

                cv2.imwrite(viz_flow_dir_ind + '/img_%05d_fwd.jpg'%(i), flow_viz.flow_to_image(fwd_flow))
                cv2.imwrite(viz_flow_dir_ind + '/img_%05d_bwd.jpg'%(i+1), flow_viz.flow_to_image(bwd_flow))

                # fwd_mask = fwd_mask * (mask_1 > 0)
                # bwd_mask = bwd_mask * (mask_2 > 0)

                masked_im2 = warped_im2.copy()
                masked_im2[mask_1==0] = 0

                masked_im0 = warped_im0.copy()
                masked_im0[mask_2==0] = 0






  
                # cv2.imwrite('./viz_warp_imgs/im_%05d.jpg'%(i), img1[..., ::-1])
                # cv2.imwrite('./viz_warp_imgs/im_%05d_fwd.jpg'%(i), warped_im2[..., ::-1])
                # cv2.imwrite('./viz_warp_imgs/im_%05d_bwd.jpg'%(i + 1), warped_im0[..., ::-1])

                cv2.imwrite(viz_warp_dir + '/im_%05d.jpg'%(i), img1[..., ::-1])
                cv2.imwrite(viz_warp_dir + '/im_%05d_fwd.jpg'%(i), masked_im2[..., ::-1])
                cv2.imwrite(viz_warp_dir + '/im_%05d_bwd.jpg'%(i + 1), masked_im0[..., ::-1 ])

            
            # if image start with 00001.png, use i+1, i+2 instead start with 00000.png then use i, i+1
            # np.savez(os.path.join(of_dir, '%05d_fwd.npz'%(i + 1)), flow=fwd_flow, mask=fwd_mask)
            # np.savez(os.path.join(of_dir, '%05d_bwd.npz'%(i + 2)), flow=bwd_flow, mask=bwd_mask)

            np.savez(os.path.join(of_dir, '%05d_fwd.npz'%(i)), flow=fwd_flow, mask=fwd_mask)
            np.savez(os.path.join(of_dir, '%05d_bwd.npz'%(i + 1)), flow=bwd_flow, mask=bwd_mask)

def run_semantic_flows(args):

    basedir = "%s"%args.data_path
    # print(basedir)
    img_dir = glob.glob(basedir + '/images_*')[0]  #basedir + '/images_*288' 

    # import pdb; pdb.set_trace() 

    # img_path_train = os.path.join(glob.glob(img_dir)[0], '%05d.png'%0)
    img_path_train = os.path.join(glob.glob(img_dir)[0], '%05d.png'%1)
    img_train = cv2.imread(img_path_train)

    # interval = 1
    interval = args.step
    of_dir = os.path.join(basedir, 'semantic_flow_i%d'%interval)
    os.makedirs(of_dir, exist_ok=True)

    if interval != 1:
        dino_tracker_dir = 'dino-tracker-i%d'%interval
    else:
        dino_tracker_dir = 'dino-tracker'

    with torch.no_grad():
        images = glob.glob(os.path.join(basedir, 'images/', '*.png')) + \
                 glob.glob(os.path.join(basedir, 'images/', '*.jpg'))
                 
        if args.step != 1: # Using step
            images = load_image_list_step(images, args.step, args.offset)
        else:
            images = load_image_list(images)

        mask_paths = sorted(glob.glob(os.path.join(basedir, 'motion_masks/', '*g')))

        # import pdb; pdb.set_trace()
            
        fwd_trajs_file = os.path.join(basedir, dino_tracker_dir, 'semantic_flow/', 'semantic_flows_i1_fwd.npy')
        fwd_trajs = np.load(fwd_trajs_file) # [N-1, H, W, 2]

        bwd_trajs_file = os.path.join(basedir, dino_tracker_dir, 'semantic_flow/', 'semantic_flows_i1_bwd.npy')
        bwd_trajs = np.load(bwd_trajs_file) # [N-1, H, W, 2]

        offset = args.offset
        MASK= True
        # if MASK:
        #     masks = sorted(glob.glob(os.path.join(basedir, 'motion_masks', '*g')))

        for i in range(fwd_trajs.shape[0]):
            print(i)
            
            image1 = images[i,None]
            image2 = images[i + 1,None]

            if image1.size(1) == 4:
                image1 = image1[:,0:3,:,:]
                image2 = image2[:,0:3,:,:]
            # print(image1.shape)
                
            # traj_idx = i - offset
                
            # if offset != 0:
            #     traj_idx = i - offset
                
            flow_up_fwd = fwd_trajs[i]
            flow_up_bwd = bwd_trajs[i]

            # flow_up_fwd = flow_up_fwd[0].cpu().numpy().transpose(1, 2, 0)
            # flow_up_bwd = flow_up_bwd[0].cpu().numpy().transpose(1, 2, 0)

            img1 = cv2.resize(np.uint8(np.clip(image1[0].cpu().numpy().transpose(1, 2, 0), 0, 255)), 
                             (img_train.shape[1], img_train.shape[0]), 
                              cv2.INTER_LINEAR)
            img2 = cv2.resize(np.uint8(np.clip(image2[0].cpu().numpy().transpose(1, 2, 0), 0, 255)), 
                             (img_train.shape[1], img_train.shape[0]), 
                             cv2.INTER_LINEAR)

            fwd_flow = resize_flow(flow_up_fwd, 
                                   img_train.shape[0], 
                                   img_train.shape[1])
            # fwd_flow = refinement_flow(fwd_flow, img1, img2)

            bwd_flow = resize_flow(flow_up_bwd, 
                                   img_train.shape[0], 
                                   img_train.shape[1])
            # bwd_flow = refinement_flow(bwd_flow, img1, img2)

            fwd_mask, bwd_mask = compute_fwdbwd_mask(fwd_flow, 
                                                     bwd_flow)
            

            mask_idx = i * interval + offset
            post_mask_idx = (i + 1) * interval + offset

            motion_mask_i = cv2.imread(mask_paths[mask_idx], cv2.IMREAD_GRAYSCALE)
            motion_mask_i = cv2.resize(motion_mask_i, 
                                       (img_train.shape[1], img_train.shape[0]), 
                                       interpolation=cv2.INTER_NEAREST)
            # 픽셀 값이 0 또는 255라고 가정 -> boolean mask로 변환
            motion_mask_i = (motion_mask_i > 128).astype(np.uint8)

            motion_mask_i1 = cv2.imread(mask_paths[post_mask_idx], cv2.IMREAD_GRAYSCALE)
            motion_mask_i1 = cv2.resize(motion_mask_i1, 
                                        (img_train.shape[1], img_train.shape[0]), 
                                        interpolation=cv2.INTER_NEAREST)
            motion_mask_i1 = (motion_mask_i1 > 128).astype(np.uint8)

            # fwd_mask, bwd_mask가 float(0/1) 또는 bool 형태라면 아래와 같이 & 연산
            # 만약 numpy.float32 등이라면 캐스팅해줘야 할 수 있음
            fwd_mask = fwd_mask.astype(np.uint8) & motion_mask_i
            bwd_mask = bwd_mask.astype(np.uint8) & motion_mask_i1
            
            if VIZ:
                viz_flow_dir = basedir + "/viz_semantic_flow_i%d"%interval
                if not os.path.exists(viz_flow_dir):
                    os.makedirs(viz_flow_dir)

                viz_warp_dir = basedir + "/viz_traj_warp_imgs_i%d"%interval
                if not os.path.exists(viz_warp_dir):
                    os.makedirs(viz_warp_dir)

                if MASK:
                    viz_warp_mask_dir = os.path.join(basedir, 'viz_warp_mask_traj')
                    os.makedirs(viz_warp_mask_dir, exist_ok=True)
                    viz_flow_dir_ind = basedir + "/viz_individiaul_semantic_flow_i%d"%interval
                    os.makedirs(viz_flow_dir_ind, exist_ok=True)    
                    
                plt.figure(figsize=(12, 6))
                plt.subplot(2,3,1)
                plt.imshow(img1)
                plt.subplot(2,3,4)
                plt.imshow(img2)

                plt.subplot(2,3,2)
                plt.imshow(flow_viz.flow_to_image(fwd_flow)/255.)
                plt.subplot(2,3,3)
                plt.imshow(flow_viz.flow_to_image(bwd_flow)/255.)

                plt.subplot(2,3,5)
                plt.imshow(flow_viz.flow_to_image(fwd_flow)/255. * np.float32(fwd_mask[..., np.newaxis]))

                plt.subplot(2,3,6)
                plt.imshow(flow_viz.flow_to_image(bwd_flow)/255. * np.float32(bwd_mask[..., np.newaxis]))

                # plt.savefig('./viz_flow/%02d.jpg'%i)
                plt.savefig(viz_flow_dir + '/%02d.jpg'%i)
                plt.close()

                warped_im2 = warp_flow(img2, fwd_flow)
                warped_im0 = warp_flow(img1, bwd_flow)

                cv2.imwrite(viz_flow_dir_ind + '/img_%05d_fwd.jpg'%(i), flow_viz.flow_to_image(fwd_flow)[..., ::-1])
                cv2.imwrite(viz_flow_dir_ind + '/img_%05d_bwd.jpg'%(i+1), flow_viz.flow_to_image(bwd_flow)[..., ::-1])
  
                # cv2.imwrite('./viz_warp_imgs/im_%05d.jpg'%(i), img1[..., ::-1])
                # cv2.imwrite('./viz_warp_imgs/im_%05d_fwd.jpg'%(i), warped_im2[..., ::-1])
                # cv2.imwrite('./viz_warp_imgs/im_%05d_bwd.jpg'%(i + 1), warped_im0[..., ::-1])

                if MASK:
                    masked_im2 = warped_im2.copy()
                    masked_im2[motion_mask_i==0] = 0
                    masked_im0 = warped_im0.copy()
                    masked_im0[motion_mask_i1==0] = 0

                    cv2.imwrite(viz_warp_mask_dir + '/im_%05d.jpg'%(i), img1[..., ::-1])
                    cv2.imwrite(viz_warp_mask_dir + '/im_%05d_fwd.jpg'%(i), masked_im2[..., ::-1])
                    cv2.imwrite(viz_warp_mask_dir + '/im_%05d_bwd.jpg'%(i + 1), masked_im0[..., ::-1 ])
                    # cv2.imwrite(viz_warp_dir + '/im_%05d.jpg'%(i), img1[..., ::-1])
                    # cv2.imwrite(viz_warp_dir + '/im_%05d_fwd.jpg'%(i), warped_im2[..., ::-1])
                    # cv2.imwrite(viz_warp_dir + '/im_%05d_bwd.jpg'%(i + 1), warped_im0[..., ::-1])
                    motion_mask_i

                cv2.imwrite(viz_warp_dir + '/im_%05d.jpg'%(i), img1[..., ::-1])
                cv2.imwrite(viz_warp_dir + '/im_%05d_fwd.jpg'%(i), warped_im2[..., ::-1])
                cv2.imwrite(viz_warp_dir + '/im_%05d_bwd.jpg'%(i + 1), warped_im0[..., ::-1])

            
            # if image start with 00001.png, use i+1, i+2 instead start with 00000.png then use i, i+1
            # np.savez(os.path.join(of_dir, '%05d_fwd.npz'%(i + 1)), flow=fwd_flow, mask=fwd_mask)
            # np.savez(os.path.join(of_dir, '%05d_bwd.npz'%(i + 2)), flow=bwd_flow, mask=bwd_mask)

            np.savez(os.path.join(of_dir, '%05d_fwd.npz'%(i)), flow=fwd_flow, mask=fwd_mask)
            np.savez(os.path.join(of_dir, '%05d_bwd.npz'%(i + 1)), flow=bwd_flow, mask=bwd_mask)

def run_semantic_flows_i(args):

    basedir = "%s"%args.data_path
    # print(basedir)
    img_dir = glob.glob(basedir + '/images_*')[0]  #basedir + '/images_*288' 

    # import pdb; pdb.set_trace() 

    # img_path_train = os.path.join(glob.glob(img_dir)[0], '%05d.png'%0)
    img_path_train = os.path.join(glob.glob(img_dir)[0], '%05d.png'%1)
    img_train = cv2.imread(img_path_train)

    # interval = 1
    interval = args.interval
    of_dir = os.path.join(basedir, 'semantic_flow_i%d'%interval)
    os.makedirs(of_dir, exist_ok=True)

    dino_tracker_dir = 'dino-tracker'

    with torch.no_grad():
        images = glob.glob(os.path.join(basedir, 'images/', '*.png')) + \
                 glob.glob(os.path.join(basedir, 'images/', '*.jpg'))
                 
        if args.step != 1: # Using step
            images = load_image_list_step(images, args.step)
        else:
            images = load_image_list(images)

        mask_paths = sorted(glob.glob(os.path.join(basedir, 'motion_masks/', '*g')))

        # import pdb; pdb.set_trace()
            
        fwd_trajs_file = os.path.join(basedir, dino_tracker_dir, 'semantic_flow/', f'semantic_flows_i{interval}_fwd.npy')
        fwd_trajs = np.load(fwd_trajs_file) # [N-1, H, W, 2]

        bwd_trajs_file = os.path.join(basedir, dino_tracker_dir, 'semantic_flow/', f'semantic_flows_i{interval}_bwd.npy')
        bwd_trajs = np.load(bwd_trajs_file) # [N-1, H, W, 2]

        for i in range(fwd_trajs.shape[0]):
            print(i)

            image1 = images[i,None]
            image2 = images[i + interval,None]

            if image1.size(1) == 4:
                image1 = image1[:,0:3,:,:]
                image2 = image2[:,0:3,:,:]
            # print(image1.shape)

            flow_up_fwd = fwd_trajs[i]
            flow_up_bwd = bwd_trajs[i]

            # flow_up_fwd = flow_up_fwd[0].cpu().numpy().transpose(1, 2, 0)
            # flow_up_bwd = flow_up_bwd[0].cpu().numpy().transpose(1, 2, 0)

            img1 = cv2.resize(np.uint8(np.clip(image1[0].cpu().numpy().transpose(1, 2, 0), 0, 255)), 
                             (img_train.shape[1], img_train.shape[0]), 
                              cv2.INTER_LINEAR)
            img2 = cv2.resize(np.uint8(np.clip(image2[0].cpu().numpy().transpose(1, 2, 0), 0, 255)), 
                             (img_train.shape[1], img_train.shape[0]), 
                             cv2.INTER_LINEAR)

            fwd_flow = resize_flow(flow_up_fwd, 
                                   img_train.shape[0], 
                                   img_train.shape[1])
            # fwd_flow = refinement_flow(fwd_flow, img1, img2)

            bwd_flow = resize_flow(flow_up_bwd, 
                                   img_train.shape[0], 
                                   img_train.shape[1])
            # bwd_flow = refinement_flow(bwd_flow, img1, img2)

            fwd_mask, bwd_mask = compute_fwdbwd_mask(fwd_flow, 
                                                     bwd_flow)
            
            mask_idx = i
            post_mask_idx = i + interval

            motion_mask_i = cv2.imread(mask_paths[mask_idx], cv2.IMREAD_GRAYSCALE)
            motion_mask_i = cv2.resize(motion_mask_i, 
                                       (img_train.shape[1], img_train.shape[0]), 
                                       interpolation=cv2.INTER_NEAREST)
            # 픽셀 값이 0 또는 255라고 가정 -> boolean mask로 변환
            motion_mask_i = (motion_mask_i > 128).astype(np.uint8)

            motion_mask_i1 = cv2.imread(mask_paths[post_mask_idx], cv2.IMREAD_GRAYSCALE)
            motion_mask_i1 = cv2.resize(motion_mask_i1, 
                                        (img_train.shape[1], img_train.shape[0]), 
                                        interpolation=cv2.INTER_NEAREST)
            motion_mask_i1 = (motion_mask_i1 > 128).astype(np.uint8)

            # fwd_mask, bwd_mask가 float(0/1) 또는 bool 형태라면 아래와 같이 & 연산
            # 만약 numpy.float32 등이라면 캐스팅해줘야 할 수 있음
            fwd_mask = fwd_mask.astype(np.uint8) & motion_mask_i
            bwd_mask = bwd_mask.astype(np.uint8) & motion_mask_i1
            
            if VIZ:
                viz_flow_dir = basedir + "/viz_semantic_flow_i%d"%interval
                if not os.path.exists(viz_flow_dir):
                    os.makedirs(viz_flow_dir)

                viz_warp_dir = basedir + "/viz_traj_warp_imgs_i%d"%interval
                if not os.path.exists(viz_warp_dir):
                    os.makedirs(viz_warp_dir)
                    
                plt.figure(figsize=(12, 6))
                plt.subplot(2,3,1)
                plt.imshow(img1)
                plt.subplot(2,3,4)
                plt.imshow(img2)

                plt.subplot(2,3,2)
                plt.imshow(flow_viz.flow_to_image(fwd_flow)/255.)
                plt.subplot(2,3,3)
                plt.imshow(flow_viz.flow_to_image(bwd_flow)/255.)

                plt.subplot(2,3,5)
                plt.imshow(flow_viz.flow_to_image(fwd_flow)/255. * np.float32(fwd_mask[..., np.newaxis]))

                plt.subplot(2,3,6)
                plt.imshow(flow_viz.flow_to_image(bwd_flow)/255. * np.float32(bwd_mask[..., np.newaxis]))

                # plt.savefig('./viz_flow/%02d.jpg'%i)
                plt.savefig(viz_flow_dir + '/%02d.jpg'%i)
                plt.close()

                warped_im2 = warp_flow(img2, fwd_flow)
                warped_im0 = warp_flow(img1, bwd_flow)
  
                # cv2.imwrite('./viz_warp_imgs/im_%05d.jpg'%(i), img1[..., ::-1])
                # cv2.imwrite('./viz_warp_imgs/im_%05d_fwd.jpg'%(i), warped_im2[..., ::-1])
                # cv2.imwrite('./viz_warp_imgs/im_%05d_bwd.jpg'%(i + 1), warped_im0[..., ::-1])

                masked_im2 = warped_im2.copy()
                masked_im2[mask_1==0] = 0

                masked_im0 = warped_im0.copy()
                masked_im0[mask_2==0] = 0
  
                # cv2.imwrite('./viz_warp_imgs/im_%05d.jpg'%(i), img1[..., ::-1])
                # cv2.imwrite('./viz_warp_imgs/im_%05d_fwd.jpg'%(i), warped_im2[..., ::-1])
                # cv2.imwrite('./viz_warp_imgs/im_%05d_bwd.jpg'%(i + 1), warped_im0[..., ::-1])

                cv2.imwrite(viz_warp_dir + '/im_%05d.jpg'%(i), img1[..., ::-1])
                cv2.imwrite(viz_warp_dir + '/im_%05d_fwd.jpg'%(i), masked_im2[..., ::-1])
                cv2.imwrite(viz_warp_dir + '/im_%05d_bwd.jpg'%(i + 1), masked_im0[..., ::-1 ])

                cv2.imwrite(viz_warp_dir + '/im_%05d.jpg'%(i), img1[..., ::-1])
                cv2.imwrite(viz_warp_dir + '/im_%05d_fwd.jpg'%(i), warped_im2[..., ::-1])
                cv2.imwrite(viz_warp_dir + '/im_%05d_bwd.jpg'%(i + interval), warped_im0[..., ::-1])

            
            # if image start with 00001.png, use i+1, i+2 instead start with 00000.png then use i, i+1
            # np.savez(os.path.join(of_dir, '%05d_fwd.npz'%(i + 1)), flow=fwd_flow, mask=fwd_mask)
            # np.savez(os.path.join(of_dir, '%05d_bwd.npz'%(i + 2)), flow=bwd_flow, mask=bwd_mask)

            np.savez(os.path.join(of_dir, '%05d_fwd.npz'%(i)), flow=fwd_flow, mask=fwd_mask)
            np.savez(os.path.join(of_dir, '%05d_bwd.npz'%(i + interval)), flow=bwd_flow, mask=bwd_mask)

def run_semantic_flows_i_all_pixel(args):

    basedir = "%s"%args.data_path
    # print(basedir)
    img_dir = glob.glob(basedir + '/images_*')[0]  #basedir + '/images_*288' 

    # import pdb; pdb.set_trace() 

    # img_path_train = os.path.join(glob.glob(img_dir)[0], '%05d.png'%0)
    img_path_train = os.path.join(glob.glob(img_dir)[0], '%05d.png'%1)
    img_train = cv2.imread(img_path_train)

    # interval = 1
    interval = args.interval
    of_dir = os.path.join(basedir, 'semantic_flow_i%d_all_pixel'%interval)
    os.makedirs(of_dir, exist_ok=True)

    dino_tracker_dir = 'dino-tracker'

    with torch.no_grad():
        images = glob.glob(os.path.join(basedir, 'images/', '*.png')) + \
                 glob.glob(os.path.join(basedir, 'images/', '*.jpg'))
                 
        if args.step != 1: # Using step
            images = load_image_list_step(images, args.step)
        else:
            images = load_image_list(images)

        mask_paths = sorted(glob.glob(os.path.join(basedir, 'motion_masks/', '*g')))

        # import pdb; pdb.set_trace()
            
        fwd_trajs_file = os.path.join(basedir, dino_tracker_dir, 'semantic_flow_all_pixel/', f'semantic_flows_i{interval}_fwd.npy')
        fwd_trajs = np.load(fwd_trajs_file) # [N-1, H, W, 2]

        bwd_trajs_file = os.path.join(basedir, dino_tracker_dir, 'semantic_flow_all_pixel/', f'semantic_flows_i{interval}_bwd.npy')
        bwd_trajs = np.load(bwd_trajs_file) # [N-1, H, W, 2]

        for i in range(fwd_trajs.shape[0]):
            print(i)
            
            image1 = images[i,None]
            image2 = images[i + interval,None]

            if image1.size(1) == 4:
                image1 = image1[:,0:3,:,:]
                image2 = image2[:,0:3,:,:]
            # print(image1.shape)
                
            flow_up_fwd = fwd_trajs[i]
            flow_up_bwd = bwd_trajs[i]

            # flow_up_fwd = flow_up_fwd[0].cpu().numpy().transpose(1, 2, 0)
            # flow_up_bwd = flow_up_bwd[0].cpu().numpy().transpose(1, 2, 0)

            img1 = cv2.resize(np.uint8(np.clip(image1[0].cpu().numpy().transpose(1, 2, 0), 0, 255)), 
                             (img_train.shape[1], img_train.shape[0]), 
                              cv2.INTER_LINEAR)
            img2 = cv2.resize(np.uint8(np.clip(image2[0].cpu().numpy().transpose(1, 2, 0), 0, 255)), 
                             (img_train.shape[1], img_train.shape[0]), 
                             cv2.INTER_LINEAR)

            fwd_flow = resize_flow(flow_up_fwd, 
                                   img_train.shape[0], 
                                   img_train.shape[1])
            # fwd_flow = refinement_flow(fwd_flow, img1, img2)

            bwd_flow = resize_flow(flow_up_bwd, 
                                   img_train.shape[0], 
                                   img_train.shape[1])
            # bwd_flow = refinement_flow(bwd_flow, img1, img2)
            BORDER=50
            fwd_flow[:BORDER, :] = 0
            fwd_flow[-BORDER:, :] = 0
            fwd_flow[:, :BORDER] = 0
            fwd_flow[:, -BORDER:] = 0

            bwd_flow[:BORDER, :] = 0
            bwd_flow[-BORDER:, :] = 0
            bwd_flow[:, :BORDER] = 0
            bwd_flow[:, -BORDER:] = 0

            fwd_mag = np.linalg.norm(fwd_flow, axis=2)  # shape: [H, W]
            mean_fwd = np.mean(fwd_mag)
            std_fwd = np.std(fwd_mag)
            thr_fwd = mean_fwd + 10 * std_fwd
            fwd_flow[fwd_mag > thr_fwd] = 0

            bwd_mag = np.linalg.norm(bwd_flow, axis=2)  # shape: [H, W]
            mean_bwd = np.mean(bwd_mag)
            std_bwd = np.std(bwd_mag)
            thr_bwd = mean_bwd + 10 * std_bwd
            bwd_flow[bwd_mag > thr_bwd] = 0

            fwd_mask, bwd_mask = compute_fwdbwd_mask(fwd_flow, 
                                                     bwd_flow)
            
            # mask_idx = i
            # post_mask_idx = i + interval

            # motion_mask_i = cv2.imread(mask_paths[mask_idx], cv2.IMREAD_GRAYSCALE)
            # motion_mask_i = cv2.resize(motion_mask_i, 
            #                            (img_train.shape[1], img_train.shape[0]), 
            #                            interpolation=cv2.INTER_NEAREST)
            # # 픽셀 값이 0 또는 255라고 가정 -> boolean mask로 변환
            # motion_mask_i = (motion_mask_i > 128).astype(np.uint8)

            # motion_mask_i1 = cv2.imread(mask_paths[post_mask_idx], cv2.IMREAD_GRAYSCALE)
            # motion_mask_i1 = cv2.resize(motion_mask_i1, 
            #                             (img_train.shape[1], img_train.shape[0]), 
            #                             interpolation=cv2.INTER_NEAREST)
            # motion_mask_i1 = (motion_mask_i1 > 128).astype(np.uint8)

            # fwd_mask, bwd_mask가 float(0/1) 또는 bool 형태라면 아래와 같이 & 연산
            # 만약 numpy.float32 등이라면 캐스팅해줘야 할 수 있음
            fwd_mask = fwd_mask.astype(np.uint8)# & motion_mask_i
            bwd_mask = bwd_mask.astype(np.uint8)# & motion_mask_i1
            
            if VIZ:
                viz_flow_dir = basedir + "/viz_semantic_flow_i%d_all_pixel"%interval
                if not os.path.exists(viz_flow_dir):
                    os.makedirs(viz_flow_dir)

                viz_warp_dir = basedir + "/viz_traj_warp_imgs_i%d_all_pixel"%interval
                if not os.path.exists(viz_warp_dir):
                    os.makedirs(viz_warp_dir)
                    
                plt.figure(figsize=(12, 6))
                plt.subplot(2,3,1)
                plt.imshow(img1)
                plt.subplot(2,3,4)
                plt.imshow(img2)

                plt.subplot(2,3,2)
                plt.imshow(flow_viz.flow_to_image(fwd_flow)/255.)
                plt.subplot(2,3,3)
                plt.imshow(flow_viz.flow_to_image(bwd_flow)/255.)

                plt.subplot(2,3,5)
                plt.imshow(flow_viz.flow_to_image(fwd_flow)/255. * np.float32(fwd_mask[..., np.newaxis]))

                plt.subplot(2,3,6)
                plt.imshow(flow_viz.flow_to_image(bwd_flow)/255. * np.float32(bwd_mask[..., np.newaxis]))

                # plt.savefig('./viz_flow/%02d.jpg'%i)
                plt.savefig(viz_flow_dir + '/%02d.jpg'%i)
                plt.close()

                warped_im2 = warp_flow(img2, fwd_flow)
                warped_im0 = warp_flow(img1, bwd_flow)
  
                # cv2.imwrite('./viz_warp_imgs/im_%05d.jpg'%(i), img1[..., ::-1])
                # cv2.imwrite('./viz_warp_imgs/im_%05d_fwd.jpg'%(i), warped_im2[..., ::-1])
                # cv2.imwrite('./viz_warp_imgs/im_%05d_bwd.jpg'%(i + 1), warped_im0[..., ::-1])

                cv2.imwrite(viz_warp_dir + '/im_%05d.jpg'%(i), img1[..., ::-1])
                cv2.imwrite(viz_warp_dir + '/im_%05d_fwd.jpg'%(i), warped_im2[..., ::-1])
                cv2.imwrite(viz_warp_dir + '/im_%05d_bwd.jpg'%(i + interval), warped_im0[..., ::-1])

            
            # if image start with 00001.png, use i+1, i+2 instead start with 00000.png then use i, i+1
            # np.savez(os.path.join(of_dir, '%05d_fwd.npz'%(i + 1)), flow=fwd_flow, mask=fwd_mask)
            # np.savez(os.path.join(of_dir, '%05d_bwd.npz'%(i + 2)), flow=bwd_flow, mask=bwd_mask)

            np.savez(os.path.join(of_dir, '%05d_fwd.npz'%(i)), flow=fwd_flow, mask=fwd_mask)
            np.savez(os.path.join(of_dir, '%05d_bwd.npz'%(i + interval)), flow=bwd_flow, mask=bwd_mask)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--small', 
                        action='store_true', 
                        help='use small model')
    parser.add_argument('--mixed_precision', 
                        action='store_true',
                        help='use mixed precision')
    parser.add_argument("--data_path", type=str, 
                        help='COLMAP Directory')
    parser.add_argument("--epi_threshold", type=float, 
                        default=1.0,
                        help='epipolar distance threshold for physical motion segmentation')
    parser.add_argument("--multi_camera", action='store_true', help='when use multi-camera setting')
    parser.add_argument("--data_type", type=str, default='colmap', help='option: colmap / blender')
    parser.add_argument("--step", type=int, default=1, help='image step')
    parser.add_argument("--skip_of", action='store_true', default=False)
    parser.add_argument("--traj_init", action='store_true', default=False)
    parser.add_argument("--skip_moseg", action='store_true', default=False)
    parser.add_argument("--semantic_flow", action='store_true', default=False)
    parser.add_argument("--semantic_flow_i", action='store_true', default=False)
    parser.add_argument("--semantic_flow_all_pixel", action='store_true', default=False)
    parser.add_argument("--interval", type=int, default=1, help='interval of trajectry flow')
    parser.add_argument("--offset", type=int, default=0, help='off set for trajcetory flow')
    parser.add_argument("--of_mask", action='store_true', default=False)
    # parser.add_argument("--input_flow_w", type=int, 
                        # default=768,
                        # help='input image width for optical flow, \
                        # the height will be computed based on original aspect ratio ')

    # parser.add_argument("--input_semantic_w", type=int, 
                        # default=1024,
                        # help='input image width for semantic segmentation')

    # parser.add_argument("--input_semantic_h", type=int, 
                        # default=576,
                        # help='input image height for semantic segmentation')
        

    args = parser.parse_args()
    if args.semantic_flow:
        run_semantic_flows(args)
        sys.exit(0)
    elif args.semantic_flow_i:
        run_semantic_flows_i(args)
        sys.exit(0)
    if not args.skip_of:
        run_optical_flows(args)
        sys.exit(0)
    if args.of_mask:
        run_optical_flows_mask(args)
        sys.exit(0)
    elif args.semantic_flow_all_pixel:
        run_semantic_flows_i_all_pixel(args)
        sys.exit(0)
    if (args.data_type == 'colmap') & (not args.skip_moseg):
        motion_segmentation(args.data_path, args.epi_threshold, args.multi_camera)
    if (args.data_type == 'blender') & (not args.skip_moseg):
        motion_segmentation_blender(args.data_path, args.epi_threshold, args.multi_camera, args.step)