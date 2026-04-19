import colmap_read_model as read_model
from blender_read_model import BlenderDataset
import numpy as np
import os
import sys
import json
from glob import glob
import cv2
import shutil

def get_bbox_corners(points):
  lower = points.min(axis=0)
  upper = points.max(axis=0)
  return np.stack([lower, upper])

def filter_outlier_points(points, inner_percentile):
  """Filters outlier points."""
  outer = 1.0 - inner_percentile
  lower = outer / 2.0
  upper = 1.0 - lower
  centers_min = np.quantile(points, lower, axis=0)
  centers_max = np.quantile(points, upper, axis=0)
  result = points.copy()

  too_near = np.any(result < centers_min[None, :], axis=1)
  too_far = np.any(result > centers_max[None, :], axis=1)

  return result[~(too_near | too_far)]

def load_colmap_data(realdir, args):
    camerasfile = os.path.join(realdir, 'sparse/cameras.bin')
    camdata = read_model.read_cameras_binary(camerasfile)
    
    list_of_keys = sorted(list(camdata.keys()))

    ### shindy modified ###
    if args.multi_camera:
      HWF = []
      for i in range(len(list_of_keys)):
        cam = camdata[list_of_keys[i]]
        h, w, f = cam.height, cam.width, cam.params[0]
        hwf = np.array([h, w, f]).reshape([3,1])
        HWF.append(hwf)
      
      HWF = np.stack(HWF, 2)

    else: # original
      cam = camdata[list_of_keys[0]]
      print( 'Cameras', len(cam))

      h, w, f = cam.height, cam.width, cam.params[0]
      # w, h, f = factor * w, factor * h, factor * f
      hwf = np.array([h, w, f]).reshape([3,1])
    # ######################
       

    # # cam = camdata[list_of_keys[0]]
    # # print( 'Cameras', len(cam))

    # # h, w, f = cam.height, cam.width, cam.params[0]
    # # # w, h, f = factor * w, factor * h, factor * f
    # # hwf = np.array([h, w, f]).reshape([3,1])

    # ### Shindy Modified ###
    # HWF = []
    # for i in range(len(list_of_keys)):
    #   cam = camdata[list_of_keys[i]]
    #   h, w, f = cam.height, cam.width, cam.params[0]
    #   hwf = np.array([h, w, f]).reshape([3,1])
    #   HWF.append(hwf)
      
    # HWF = np.stack(HWF, 2)
    # ######################

    imagesfile = os.path.join(realdir, 'sparse/images.bin')
    imdata = read_model.read_images_binary(imagesfile)
    
    w2c_mats = []
    bottom = np.array([0,0,0,1.]).reshape([1,4])
    
    names = [imdata[k].name for k in imdata]
    img_keys = [k for k in imdata]

    print( 'Images #', len(names))
    perm = np.argsort(names)

    points3dfile = os.path.join(realdir, 'sparse/points3D.bin')
    pts3d = read_model.read_points3d_binary(points3dfile)

    # extract point 3D xyz
    point_cloud = []
    for key in pts3d:
        point_cloud.append(pts3d[key].xyz)

    point_cloud = np.stack(point_cloud, 0)
    point_cloud = filter_outlier_points(point_cloud, 0.95)

    bounds_mats = []

    upper_bound = 1000
    
    if upper_bound < len(img_keys):
        print("Only keeping " + str(upper_bound) + " images!")

    for i in perm[0:min(upper_bound, len(img_keys))]:
        im = imdata[img_keys[i]]
        print(im.name)
        R = im.qvec2rotmat()
        t = im.tvec.reshape([3,1])
        m = np.concatenate([np.concatenate([R, t], 1), bottom], 0)
        w2c_mats.append(m)

        pts_3d_idx = im.point3D_ids
        pts_3d_vis_idx = pts_3d_idx[pts_3d_idx >= 0]

        # 
        depth_list = []
        for k in range(len(pts_3d_vis_idx)):
          point_info = pts3d[pts_3d_vis_idx[k]]
          P_g = point_info.xyz
          P_c = np.dot(R, P_g.reshape(3, 1)) + t.reshape(3, 1)
          depth_list.append(P_c[2])

        zs = np.array(depth_list)
        close_depth, inf_depth = np.percentile(zs, 5), np.percentile(zs, 95)
        bounds = np.array([close_depth, inf_depth])
        bounds_mats.append(bounds)

    w2c_mats = np.stack(w2c_mats, 0)
    # bounds_mats = np.stack(bounds_mats, 0)
    c2w_mats = np.linalg.inv(w2c_mats)

    viz = True
    if viz == True:

        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        positions = c2w_mats[:, :, 3]

        # Plotting in 3D
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')

        # Plot the camera positions
        ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c='r', marker='o', label='Camera ref Positions')

        # Setting labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Visualization of Camera Poses')

        # Adding a legend
        ax.legend()

        plt.show()

    # bbox_corners = get_bbox_corners(point_cloud)
    # also add camera 
    bbox_corners = get_bbox_corners(
                    np.concatenate([point_cloud, c2w_mats[:, :3, 3]], axis=0))
    
    scene_center = np.mean(bbox_corners, axis=0)
    scene_scale = 1.0 / np.sqrt(np.sum((bbox_corners[1] - bbox_corners[0]) ** 2))

    print('bbox_corners ', bbox_corners)
    print('scene_center ', scene_center, scene_scale)


    poses = c2w_mats[:, :3, :4].transpose([1,2,0])  # [3, 4, N]
    ### Shindy Modified ###
    if args.multi_camera:
      # import pdb; pdb.set_trace()
      ### concat all camera HWF to posese
      poses = np.concatenate([poses, HWF], 1)
    else: # original
      poses = np.concatenate([poses, np.tile(hwf[..., np.newaxis], 
                                          [1,1,poses.shape[-1]])], 1) # [3, 5, N]
    ########################

    # must switch to [-y, x, z] from [x, -y, -z], NOT [r, u, -t]
    poses = np.concatenate([poses[:, 1:2, :], poses[:, 0:1, :], 
                            -poses[:, 2:3, :], 
                            poses[:, 3:4, :], 
                            poses[:, 4:5, :]], 1)
    
    save_arr = []

    for i in range((poses.shape[2])):
        save_arr.append(np.concatenate([poses[..., i].ravel(), bounds_mats[i]], 0))

    save_arr = np.array(save_arr)
    print(save_arr.shape)
    np.save(os.path.join(realdir, 'poses_bounds.npy'), save_arr)
    with open(os.path.join(realdir, 'scene.json'), 'w') as f:
      json.dump({
          'scale': scene_scale,
          'center': scene_center.tolist(),
          'bbox': bbox_corners.tolist(),
      }, f, indent=2)

def load_colmap_data_3cam(realdir, args):
    camerasfile = os.path.join(realdir, 'sparse/cameras.bin')
    camdata = read_model.read_cameras_binary(camerasfile)
    
    list_of_keys = sorted(list(camdata.keys()))

    ### shindy modified ###
    HWF = []
    for i in range(len(list_of_keys)):
      cam = camdata[list_of_keys[i]]
      h, w, f = cam.height, cam.width, cam.params[0]
      hwf = np.array([h, w, f]).reshape([3,1])
      HWF.append(hwf)
    
    HWF = np.stack(HWF, 2)

    imagesfile = os.path.join(realdir, 'sparse/images.bin')
    imdata = read_model.read_images_binary(imagesfile)
    
    w2c_mats = []
    bottom = np.array([0,0,0,1.]).reshape([1,4])
    
    names = [imdata[k].name for k in imdata]
    img_keys = [k for k in imdata]

    print( 'Images #', len(names))
    perm = np.argsort(names)

    points3dfile = os.path.join(realdir, 'sparse/points3D.bin')
    pts3d = read_model.read_points3d_binary(points3dfile)

    # extract point 3D xyz
    point_cloud = []
    for key in pts3d:
        point_cloud.append(pts3d[key].xyz)

    point_cloud = np.stack(point_cloud, 0)
    point_cloud = filter_outlier_points(point_cloud, 0.95)

    bounds_mats = []

    upper_bound = 1000
    
    if upper_bound < len(img_keys):
        print("Only keeping " + str(upper_bound) + " images!")

    for i in perm[0:min(upper_bound, len(img_keys))]:
        im = imdata[img_keys[i]]
        print(im.name)
        R = im.qvec2rotmat()
        t = im.tvec.reshape([3,1])
        m = np.concatenate([np.concatenate([R, t], 1), bottom], 0)
        w2c_mats.append(m)

        pts_3d_idx = im.point3D_ids
        pts_3d_vis_idx = pts_3d_idx[pts_3d_idx >= 0]

        # 
        depth_list = []
        for k in range(len(pts_3d_vis_idx)):
          point_info = pts3d[pts_3d_vis_idx[k]]
          P_g = point_info.xyz
          P_c = np.dot(R, P_g.reshape(3, 1)) + t.reshape(3, 1)
          depth_list.append(P_c[2])

        zs = np.array(depth_list)
        close_depth, inf_depth = np.percentile(zs, 5), np.percentile(zs, 95)
        bounds = np.array([close_depth, inf_depth])
        bounds_mats.append(bounds)

    w2c_mats = np.stack(w2c_mats, 0)
    # bounds_mats = np.stack(bounds_mats, 0)
    c2w_mats = np.linalg.inv(w2c_mats)

    # bbox_corners = get_bbox_corners(point_cloud)
    # also add camera 
    bbox_corners = get_bbox_corners(
                    np.concatenate([point_cloud, c2w_mats[:, :3, 3]], axis=0))
    
    scene_center = np.mean(bbox_corners, axis=0)
    scene_scale = 1.0 / np.sqrt(np.sum((bbox_corners[1] - bbox_corners[0]) ** 2))

    print('bbox_corners ', bbox_corners)
    print('scene_center ', scene_center, scene_scale)


    poses = c2w_mats[:, :3, :4].transpose([1,2,0])  # [3, 4, N]
    ### concat all camera HWF to posese
    if HWF.shape[-1] != poses.shape[-1]:
      HWF = np.tile(HWF, [1, 1, int(poses.shape[-1]/3)])
    
    poses = np.concatenate([poses, HWF], 1)


    # must switch to [-y, x, z] from [x, -y, -z], NOT [r, u, -t]
    poses = np.concatenate([poses[:, 1:2, :], poses[:, 0:1, :], 
                            -poses[:, 2:3, :], 
                            poses[:, 3:4, :], 
                            poses[:, 4:5, :]], 1)
    
    save_arr = []

    for i in range((poses.shape[2])):
        save_arr.append(np.concatenate([poses[..., i].ravel(), bounds_mats[i]], 0))

    save_arr = np.array(save_arr)
    # If 3 camera setting
    ### ------------------------------------
    high_save_arr = save_arr[0::3, ...]
    low_save_arr = save_arr[1::3, ...]
    mid_save_arr = save_arr[2::3, ...]
    
    temp_arr = np.zeros([mid_save_arr.shape[0], 17])
    
    for i in range(mid_save_arr.shape[0]):
      if i % 3 == 0:
        temp_arr[i] = mid_save_arr[i]
      elif i % 3 == 1:
        temp_arr[i] = high_save_arr[i]
      else:
        temp_arr[i] = low_save_arr[i]
    save_arr = temp_arr
    ### ------------------------------------
    print(save_arr.shape)
    np.save(os.path.join(realdir, 'poses_bounds.npy'), save_arr)
    with open(os.path.join(realdir, 'scene.json'), 'w') as f:
      json.dump({
          'scale': scene_scale,
          'center': scene_center.tolist(),
          'bbox': bbox_corners.tolist(),
      }, f, indent=2)
      
      
    
    import pdb; pdb.set_trace()
    image_files = sorted(glob(os.path.join(realdir, 'images', '*g')))
    output_dir = os.path.join(realdir, 'images_')
    os.makedirs(output_dir, exist_ok=True)
    
    high_image_files = image_files[0::3]
    low_image_files = image_files[1::3]
    mid_image_files = image_files[2::3]
    
    for i in range( int(len(image_files)/3) ):
      image = cv2.imread(image_files[i])
      
      if i % 3 == 0:
        image = mid_image_files[i]
      elif i % 3 == 1:
        image = high_image_files[i]
      else:
        image = low_image_files[i]
        
      new_name = str(int(i)).zfill(5) + ".png"
      output_path = os.path.join(output_dir, new_name)
      shutil.copy(image, output_path)
      
    
    
    original_image_dir = os.path.join(realdir, 'images')
    new_name_original_image_dir = os.path.join(realdir, 'original_images')
      
    shutil.move(original_image_dir, new_name_original_image_dir)
    shutil.move(output_dir, original_image_dir)
      
def load_blender_data(realdir, args):
    train_dataset = BlenderDataset(
        realdir,
        "train",
        1.0
    )

    # import pdb; pdb.set_trace()
    
    c2w_mats = train_dataset.poses.detach().cpu().numpy()

    w, h = train_dataset.img_wh
    f = train_dataset.focal
    hwf = np.array([h, w, f]).reshape([3,1])

    poses = c2w_mats[:, :3, :4].transpose([1,2,0])  # [3, 4, N]
    poses = np.concatenate([poses, np.tile(hwf[..., np.newaxis], [1,1,poses.shape[-1]])], 1) # [3, 5, N]

    # poses = np.concatenate([poses[:, 1:2, :], poses[:, 0:1, :], 
    #                         -poses[:, 2:3, :], 
    #                         poses[:, 3:4, :], 
    #                         poses[:, 4:5, :]], 1)
    
    bounds_mat = train_dataset.near_far

    save_arr = []
    
    for i in range((poses.shape[2])):
        save_arr.append(np.concatenate([poses[..., i].ravel(), bounds_mat], 0))

    save_arr = np.array(save_arr)
    print(save_arr.shape)
    np.save(os.path.join(realdir, 'poses_bounds.npy'), save_arr)

    # save_arr = []

    # for i in range((poses.shape[2])):
    #     save_arr.append(np.concatenate([poses[..., i].ravel(), bounds_mats[i]], 0))
    
    # save_arr = np.array(save_arr)
    # print(save_arr.shape)
    # np.save(os.path.join(realdir, 'poses_bounds.npy'), save_arr)





   

  

import argparse

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, 
                        help='COLMAP Directory')
    parser.add_argument("--multi_camera", action='store_true', help='when use multi-camera setting')
    parser.add_argument("--data_type", type=str, default='colmap', help='options: colmap / blender')

    args = parser.parse_args()

    basedir = args.data_path #"/phoenix/S7/zl548/nerf_data/%s/dense"%scene_name

    if args.data_type == 'blender':
       load_blender_data(basedir, args)
    elif args.data_type == '3cam':
      load_colmap_data_3cam(basedir, args)
    else:
       load_colmap_data(basedir, args)

    print( 'Done with imgs2poses' )
