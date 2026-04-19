import numpy as np
import os
import sys
import glob
import cv2
import scipy.io

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def read_img(img_dir, img1_name, img2_name):
  # print(os.path.join(img_dir, img1_name + '.png'))
  return cv2.imread(os.path.join(img_dir, img1_name + '.png')), cv2.imread(os.path.join(img_dir, img2_name + '.png'))

def refinement_flow(fwd_flow, img1, img2):
  flow_refine = cv2.VariationalRefinement.create()

  refine_flow = flow_refine.calc(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), 
                                 cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY), 
                                 fwd_flow)
  
  return refine_flow

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


def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow_new = flow.copy()
    flow_new[:,:,0] += np.arange(w)
    flow_new[:,:,1] += np.arange(h)[:,np.newaxis]

    res = cv2.remap(img, flow_new, None, cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
    return res

def resize_flow(flow, img_h, img_w):
    # flow = np.load(flow_path)
    # flow_h, flow_w = flow.shape[0], flow.shape[1]
    flow[:, :, 0] *= float(img_w)/float(flow_w)
    flow[:, :, 1] *= float(img_h)/float(flow_h)
    flow = cv2.resize(flow, (img_w, img_h), cv2.INTER_LINEAR)

    return flow

def extract_poses(im):
    R = im.qvec2rotmat()
    t = im.tvec.reshape([3,1])
    bottom = np.array([0,0,0,1.]).reshape([1,4])

    m = np.concatenate([np.concatenate([R, t], 1), bottom], 0)

    return m

def load_colmap_data(realdir):
    import colmap_read_model as read_model

    camerasfile = os.path.join(realdir, 'sparse/cameras.bin')
    camdata = read_model.read_cameras_binary(camerasfile)
    
    list_of_keys = list(camdata.keys())
    cam = camdata[list_of_keys[0]]
    print( 'Cameras', len(cam))

    h, w, f = cam.height, cam.width, cam.params[0]
    # w, h, f = factor * w, factor * h, factor * f
    hwf = np.array([h,w,f]).reshape([3,1])

    imagesfile = os.path.join(realdir, 'sparse/images.bin')
    imdata = read_model.read_images_binary(imagesfile)
    
    w2c_mats = []
    # bottom = np.array([0,0,0,1.]).reshape([1,4])
    
    names = [imdata[k].name for k in imdata]
    img_keys = [k for k in imdata]

    print( 'Images #', len(names))
    perm = np.argsort(names)

    return imdata, perm, img_keys, hwf

def load_blender_data(realdir):
    from blender_read_model import BlenderDataset

    train_dataset = BlenderDataset(
        realdir,
        "train",
        1.0
    )

    w, h = train_dataset.img_wh
    f = train_dataset.focal
    hwf = np.array([h, w, f]).reshape([3,1])


    # camerasfile = os.path.join(realdir, 'sparse/cameras.bin')
    # camdata = read_model.read_cameras_binary(camerasfile)
    
    # list_of_keys = list(camdata.keys())
    # cam = camdata[list_of_keys[0]]
    # print( 'Cameras', len(cam))

    # h, w, f = cam.height, cam.width, cam.params[0]
    # # w, h, f = factor * w, factor * h, factor * f
    # hwf = np.array([h,w,f]).reshape([3,1])

    # imagesfile = os.path.join(realdir, 'sparse/images.bin')
    # imdata = read_model.read_images_binary(imagesfile)
    
    # w2c_mats = []
    # # bottom = np.array([0,0,0,1.]).reshape([1,4])
    
    # names = [imdata[k].name for k in imdata]
    # img_keys = [k for k in imdata]

    # print( 'Images #', len(names))
    # perm = np.argsort(names)

    return imdata, perm, img_keys, hwf

def load_colmap_data_multi_focal(realdir):
    import colmap_read_model as read_model

    camerasfile = os.path.join(realdir, 'sparse/cameras.bin')
    camdata = read_model.read_cameras_binary(camerasfile)
    
    list_of_keys = list(camdata.keys())

    HWF = []
    for i in range(len(list_of_keys)):
      cam = camdata[list_of_keys[i]]
      h, w, f = cam.height, cam.width, cam.params[0]
      hwf = np.array([h, w, f]).reshape([3,1])
      HWF.append(hwf)

    HWF = np.stack(HWF, 2)

    # cam = camdata[list_of_keys[0]]
    # print( 'Cameras', len(cam))

    # h, w, f = cam.height, cam.width, cam.params[0]
    # # w, h, f = factor * w, factor * h, factor * f
    # hwf = np.array([h,w,f]).reshape([3,1])

    imagesfile = os.path.join(realdir, 'sparse/images.bin')
    imdata = read_model.read_images_binary(imagesfile)
    
    w2c_mats = []
    # bottom = np.array([0,0,0,1.]).reshape([1,4])
    
    names = [imdata[k].name for k in imdata]
    img_keys = [k for k in imdata]

    print( 'Images #', len(names))
    perm = np.argsort(names)

    return imdata, perm, img_keys, HWF

def skew(x):
    return np.array([[0, -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]])


def compute_epipolar_distance(T_21, K, p_1, p_2):
    R_21 = T_21[:3, :3]
    t_21 = T_21[:3, 3]

    E_mat = np.dot(skew(t_21), R_21)
    # compute bearing vector
    inv_K = np.linalg.inv(K)

    F_mat = np.dot(np.dot(inv_K.T, E_mat), inv_K)

    l_2 = np.dot(F_mat, p_1)
    algebric_e_distance = np.sum(p_2 * l_2, axis=0)
    n_term = np.sqrt(l_2[0, :]**2 + l_2[1, :]**2) + 1e-8
    geometric_e_distance = algebric_e_distance/n_term
    geometric_e_distance = np.abs(geometric_e_distance)

    return geometric_e_distance
  
def compute_F_mat(T_21, K):
    R_21 = T_21[:3, :3]
    t_21 = T_21[:3, 3]

    E_mat = np.dot(skew(t_21), R_21)
    # compute bearing vector
    inv_K = np.linalg.inv(K)

    F_mat = np.dot(np.dot(inv_K.T, E_mat), inv_K)

    return F_mat
  
def plot_epipolar_line(F, p_1, img_shape):
    
    # Crate a grid of x-coordinates to plot the line across the image
    x_vals = np.array([0, img_shape[1]])
    
    # Line equation: ax+by+c=0 => y = -(c+ax) / b
    line_coeffs = np.dot(F, p_1)
    a, b, c = line_coeffs[0], line_coeffs[1], line_coeffs[2]
    y_vals = (-c - a * x_vals) / b
    
    plt.plot(x_vals, y_vals, 'r-', label='Epipolar Line')
    
def plot_epipolar_lines(F, img, points_1, color):
    """Plot epiploar lines for a list of points"""
    # import pdb; pdb.set_trace()
    for p1 in points_1:
      l = np.dot(F, p1)
      
      a, b, c = l
      x_vals = np.array([0, img.shape[1]])
      y_vals = (-c -a*x_vals)/b
      
      img = cv2.line(img, (int(x_vals[0]), int(y_vals[0])), (int(x_vals[1]), int(y_vals[1])), color, 1)      
    return img
      
def visualize_epiploar_on_images(T_21, K, img1, img2, points_1, points_2, i):
  
    F = compute_F_mat(T_21, K)
    
    # Create color array for drawing lines
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
    
    # Draw epipolar lines on both images
    # img1_with_lines = plot_epipolar_lines(F.T, img1, points_2, colors[0])
    img2_with_lines = plot_epipolar_lines(F, img2, points_1, colors[1])
    
    for p2 in points_2:
      img2_with_lines = cv2.circle(img2_with_lines, (int(p2[0]), int(p2[1])), radius=5, color=colors[0])
      
    img1_with_points = img1
    for p1 in points_1:
      img1_with_points = cv2.circle(img1_with_points, (int(p1[0]), int(p1[1])), radius=5, color=colors[0])
    
    
    # img1_with_points = cv2.circle(img1, (int(points_1[0]), int(points_1[0])), radius=0, color=colors[0])
    
    # Concatenate images side by side
    # combined_image = np.hstack((img1_with_lines, img2_with_lines))
    
    # Display images
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    # axes[0].imshow(cv2.cvtColor(img1_with_lines, cv2.COLOR_BGR2RGB))
    axes[0].imshow(cv2.cvtColor(img1_with_points, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Epipolar Lines on Image 1')
    axes[0].axis('off')

    axes[1].imshow(cv2.cvtColor(img2_with_lines, cv2.COLOR_BGR2RGB))
    axes[1].set_title('Epipolar Lines on Image 2')
    axes[1].axis('off')
    
    plt.savefig(f'./epipolar/epiploar_geometry_{i:05d}.png')
    plt.show()
    
  
  
def visualize_epipolar_geometry(T_21, K, p_1, p_2, img_shape, i):
    F_mat = compute_F_mat(T_21, K)
    
    plt.figure(figsize=(8,6))
    plt.xlim(0, img_shape[1])
    plt.ylim(img_shape[0], 0)
    
    # Plot epipolar line in the second image
    plot_epipolar_line(F_mat, p_1, img_shape)
    
    # Plot point p_2 in the second image
    plt.plot(p_2[0], p_2[1], 'bo', label='Point in Image 2')
    
    plt.title('Epiploar line and Corresponding Point')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    
    plt.savefig(f'./epipolar/epiploar_geometry_{i:05d}.png')
    plt.show()    
    

def compute_epipolar_distance_multi_focus(T_21, K1, K2, p_1, p_2):
    R_21 = T_21[:3, :3]
    t_21 = T_21[:3, 3]

    E_mat = np.dot(skew(t_21), R_21)
    # compute bearing vector
    inv_K1 = np.linalg.inv(K1)
    inv_K2 = np.linalg.inv(K2)

    F_mat = np.dot(np.dot(inv_K2.T, E_mat), inv_K1)

    l_2 = np.dot(F_mat, p_1)
    algebric_e_distance = np.sum(p_2 * l_2, axis=0)
    n_term = np.sqrt(l_2[0, :]**2 + l_2[1, :]**2) + 1e-8
    geometric_e_distance = algebric_e_distance/n_term
    geometric_e_distance = np.abs(geometric_e_distance)

    return geometric_e_distance

def read_optical_flow(basedir, img_i_name, read_fwd):
    flow_dir = os.path.join(basedir, 'flow_i1')

    fwd_flow_path = os.path.join(flow_dir, '%s_fwd.npz'%img_i_name[:-4])
    bwd_flow_path = os.path.join(flow_dir, '%s_bwd.npz'%img_i_name[:-4])
    
    # fwd_flow_path = os.path.join(flow_dir, '%s_fwd.npz'%img_i_name)
    # bwd_flow_path = os.path.join(flow_dir, '%s_bwd.npz'%img_i_name)

    if read_fwd:
      fwd_data = np.load(fwd_flow_path)#, (w, h))
      fwd_flow, fwd_mask = fwd_data['flow'], fwd_data['mask']
      # fwd_mask = np.float32(fwd_mask)

      # bwd_flow = np.zeros_like(fwd_flow)
      return fwd_flow, fwd_mask
    else:
      bwd_data = np.load(bwd_flow_path)#, (w, h))
      bwd_flow, bwd_mask = bwd_data['flow'], bwd_data['mask']
      # bwd_mask = np.float32(bwd_mask)
      # fwd_flow = np.zeros_like(bwd_flow)
      return bwd_flow, bwd_mask
    # return fwd_flow, bwd_flow#, fwd_mask, bwd_mask
