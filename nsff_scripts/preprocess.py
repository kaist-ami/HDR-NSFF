import shutil
import os
import argparse
from glob import glob
from PIL import Image
import cv2
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from pathlib import Path

def resize_image(input_path, output_path, target_width):
    with Image.open(input_path) as img:
        # Calculate the target height to maintpyain the aspect ratio
        width_percent = (target_width / float(img.size[0]))
        target_height = int((float(img.size[1]) * float(width_percent)))

        # Resize the image
        img = img.resize((target_width, target_height), Image.LANCZOS)
        
        # Save the resized image
        img.save(output_path)


def rename_images(args):
    source_dir = args.src_dir 
    destination_dir = os.path.join(args.src_dir, 'images')
    os.makedirs(destination_dir, exist_ok=True)

    # if not os.path.exists(keep_dir):
    #     os.makedirs(keep_dir)

    img_list = sorted(glob(os.path.join(source_dir, '*.JPG')))
    print(img_list)
    
    idx = 0
    for img in img_list:
        # img_dir, img_name = os.path.split(img)
        
        # keep_img_path = os.path.join(keep_dir, img_name)

        new_img_name = str(int(idx)).zfill(5) + ".png"
        new_img_path = os.path.join(destination_dir, new_img_name)
        shutil.copy(img, new_img_path)

        resize_image(new_img_path, new_img_path, 1280)
        # shutil.move(img, keep_img_path)
        idx += 1

    return

def sample_images():
    source_dir = "/Users/shindy/Projects/hdr-4d/dataset/blender/tank/v1/"
    
    destination_dir = os.path.join(source_dir, "mid_step_60_108_2")
    # keep_dir = source_dir + "/origin"

    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    # if not os.path.exists(keep_dir):
    #     os.makedirs(keep_dir)

    img_list = sorted(glob(os.path.join(source_dir, 'mid', '*.png')))
    print(img_list)
    
    new_name_idx = 1
    img_idx = 0
    for _ in range(24):
        # img_dir, img_name = os.path.split(img)
        
        # keep_img_path = os.path.join(keep_dir, img_name)
        img = img_list[img_idx]

        new_img_name = str(int(new_name_idx)).zfill(5) + ".png"
        new_img_path = os.path.join(destination_dir, new_img_name)
        shutil.copy(img, new_img_path)

        # resize_image(new_img_path, new_img_path, 720)
        # shutil.move(img, keep_img_path)
        new_name_idx += 1
        img_idx += 2

    return

def extract_frames(video_path, output_dir):
    cap = cv2.VideoCapture(video_path)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_filename = os.path.join(output_dir, f'frame_{frame_count:05d}.jpg')
        cv2.imwrite(frame_filename, frame)
        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

def extract_videos2frames(videos_dir):

    video_list = os.listdir(videos_dir)
    for video_name in video_list:
        video_path = os.path.join(videos_dir, video_name)
        output_path = os.path.join(videos_dir, f"../frames/{video_name.split('.')[0]}")
        extract_frames(video_path, output_path)

def rename_sort(video_frames_dir):

    video_list = os.listdir(video_frames_dir)

    for video_name in video_list:
        video_path = os.path.join(video_frames_dir, video_name)
        frame_list = sorted(glob(os.path.join(video_path, '*g')))
        for idx, frame in enumerate(frame_list):
            
            new_frame_name = str(int(idx)).zfill(5) + ".png"
            new_frame_path = os.path.join(video_path, new_frame_name)
            shutil.move(frame, new_frame_path)

def mix_3_images():
    mid_dir = "/node_data/shindy/projects/hdr-4d/NSFF/dataset/blender-lego/v2_mid/origin"
    high_dir = "/node_data/shindy/projects/hdr-4d/NSFF/dataset/blender-lego/v7/EV7"
    low_dir = "/node_data/shindy/projects/hdr-4d/NSFF/dataset/blender-lego/v7/EV14"

    output_dir = "/node_data/shindy/projects/hdr-4d/NSFF/dataset/blender-lego/v7/mix3"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    mid_images = sorted(glob(os.path.join(mid_dir, "*.png")))
    high_images = sorted(glob(os.path.join(high_dir, "*.png")))
    low_images = sorted(glob(os.path.join(low_dir, "*.png")))

    for idx in range(1,len(mid_images)):
        print(idx)
        
        remainder = idx % 3
        if remainder == 1:
            image = mid_images[idx]
        elif remainder == 2:
            image = high_images[idx // 3]
        else:
            image = low_images[idx // 3 - 1]

        print(image)

        new_image_name = str(int(idx)).zfill(5) + ".png"
        new_image_path = os.path.join(output_dir, new_image_name)
        shutil.copy(image, new_image_path)

    return

def mix_frame(src_dir):
    mid_dir = os.path.join(src_dir, 'mid')
    high_dir = os.path.join(src_dir, 'high')
    low_dir = os.path.join(src_dir, 'low')

    output_dir = os.path.join(src_dir, 'mix', 'images')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    mid_frame_list = sorted(glob(os.path.join(mid_dir, '*g')))
    high_frame_list = sorted(glob(os.path.join(high_dir, '*g')))
    low_frame_list = sorted(glob(os.path.join(low_dir, '*g')))

    for idx in range(len(mid_frame_list)):
        
        r = idx % 3
        if r == 0:
            frame = mid_frame_list[idx]
        elif r == 1:
            frame = high_frame_list[idx]
        else:
            frame = low_frame_list[idx]

        print(f'idx: {idx+1}, frame: {frame}')

        new_frame_name = str(int(idx+1)).zfill(5) + ".png"
        new_frame_path = os.path.join(output_dir, new_frame_name)
        shutil.copy(frame, new_frame_path)

    return


def center_crop(src_dir):
    # Check if the directory exists
    if not os.path.exists(src_dir):
        print(f"Directory {src_dir} does not exist.")
        return
    
    # Get list of all image files in the directory
    image_files = [f for f in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, f)) and f.lower().endswith(('png', 'jpg', 'jpeg'))]
    
    # Find the smallest resolution among all images
    min_width, min_height = float('inf'), float('inf')
    for image_file in image_files:
        with Image.open(os.path.join(src_dir, image_file)) as img:
            width, height = img.size
            if width < min_width:
                min_width = width
            if height < min_height:
                min_height = height
    
    print(f"Minimum width: {min_width}, Minimum height: {min_height}")
    
    img0_file = os.path.join(src_dir, image_files[0])
    img0 = cv2.imread(img0_file)
    H, W, _ = img0.shape
    
    if W == min_width and H == min_height:
        print('Doesn\'t need center crop')
        return
    
    # Function to center crop an image
    def crop_center(image, crop_width, crop_height):
        img_width, img_height = image.size
        left = math.floor((img_width - crop_width) / 2)
        top = math.floor((img_height - crop_height) / 2)
        right = math.floor((img_width + crop_width) / 2)
        bottom = math.floor((img_height + crop_height) / 2)
        return image.crop((left, top, right, bottom))
    
    # Crop all images to the smallest resolution

    output_path = os.path.join(src_dir, '../center_cropped/images')

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for image_file in image_files:
        print(f'Processing file: {image_file} ... ')
        with Image.open(os.path.join(src_dir, image_file)) as img:
            cropped_img = crop_center(img, min_width, min_height)
            cropped_img.save(os.path.join(output_path, f"{image_file}"))
    
    print(f"All images have been center cropped to {min_width}x{min_height}.")

def mix_frame_3(src_dir):
    mid_dir = os.path.join(src_dir, 'mid')
    high_dir = os.path.join(src_dir, 'high')
    low_dir = os.path.join(src_dir, 'low')

    output_dir = os.path.join(src_dir, 'mix_3', 'images')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    mid_frame_list = sorted(glob(os.path.join(mid_dir, '*g')))
    high_frame_list = sorted(glob(os.path.join(high_dir, '*g')))
    low_frame_list = sorted(glob(os.path.join(low_dir, '*g')))

    for idx in range(len(mid_frame_list)):
        
        r = idx % 4
        if r == 0:
            frame = mid_frame_list[idx]
        elif r == 1 or r == 3:
            frame = high_frame_list[idx]
        elif r == 2:
            frame = low_frame_list[idx]

        print(f'idx: {idx}, frame: {frame}')

        new_frame_name = str(int(idx)).zfill(5) + ".png"
        new_frame_path = os.path.join(output_dir, new_frame_name)
        shutil.copy(frame, new_frame_path)

    return

def mix_frame_2(src_dir):
    mid_dir = os.path.join(src_dir, 'mid')
    high_dir = os.path.join(src_dir, 'high')
    # low_dir = os.path.join(src_dir, 'low')

    output_dir = os.path.join(src_dir, 'mix_mid_high_2', 'images')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    mid_frame_list = sorted(glob(os.path.join(mid_dir, '*g')))
    high_frame_list = sorted(glob(os.path.join(high_dir, '*g')))
    # low_frame_list = sorted(glob(os.path.join(low_dir, '*g')))

    i=0

    for idx in range(len(mid_frame_list)):
        
        r = idx % 4
        if r == 0:
            frame = mid_frame_list[idx]
        elif r == 2:
            frame = high_frame_list[idx]
        else:
            continue

        print(f'idx: {i}, frame: {frame}')

        new_frame_name = str(int(i)).zfill(5) + ".png"
        new_frame_path = os.path.join(output_dir, new_frame_name)
        shutil.copy(frame, new_frame_path)
        i+=1

    return

def resize_images(src_dir):
    
    img_list=sorted(glob(os.path.join(src_dir, '*g')))
    for i, img in enumerate(img_list):

        resize_image(img, img, 1600)
        print(i)

    return

def mix_t(src_dir):

    mid_dir = os.path.join(src_dir, 'mid')
    high_dir = os.path.join(src_dir, 'high')
    low_dir = os.path.join(src_dir, 'low')

    output_dir = os.path.join(src_dir, 'mix_reverse', 'images')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    mid_list = sorted(glob(os.path.join(mid_dir, '*g')))
    high_list = sorted(glob(os.path.join(high_dir, '*g')), reverse=True)
    low_list = sorted(glob(os.path.join(low_dir, '*g')))

    img_list = mid_list + high_list + low_list

    for i, img in enumerate(img_list):
        print(f'idx: {i}, img: {img}')
        new_name = str(int(i)).zfill(5) + ".png"
        new_path = os.path.join(output_dir, new_name)
        shutil.copy(img, new_path)
    
    return

def mix_12321(src_dir):

    mid_dir = os.path.join(src_dir, 'mid')
    high_dir = os.path.join(src_dir, 'high')
    low_dir = os.path.join(src_dir, 'low')

    output_dir = os.path.join(src_dir, 'mix_12321', 'images')
    os.makedirs(output_dir, exist_ok=True)

    mid_list = sorted(glob(os.path.join(mid_dir, '*g')))
    high_list = sorted(glob(os.path.join(high_dir, '*g')))
    low_list = sorted(glob(os.path.join(low_dir, '*g')))
    
    img_list = []
    
    for i in range(len(mid_list)):
        
        if i % 6 == 0:
            i_img = mid_list[i]
        elif i % 6 == 1:
            i_img = high_list[i]
        elif i % 6 == 2:
            i_img = low_list[i]
        elif i % 6 == 3:
            i_img = low_list[i]
        elif i % 6 == 4:
            i_img = high_list[i]
        else:
            i_img = mid_list[i]

        img_list.append(i_img)

    for i, img in enumerate(img_list):
        print(f'idx: {i}, img: {img}')
        new_name = str(int(i)).zfill(5) + ".png"
        new_path = os.path.join(output_dir, new_name)
        shutil.copy(img, new_path)
    
    return

def video2images(src_dir):

    video = (glob(os.path.join(src_dir, '*.MOV')) \
             + glob(os.path.join(src_dir, '*.mp4')) + glob(os.path.join(src_dir, '*.mov')) )[0]

    image_dir = os.path.join(src_dir, 'converted_images')
    os.makedirs(image_dir, exist_ok=True)
    
    vidcap = cv2.VideoCapture(video)

    success, image = vidcap.read()
    count = 0
    while success:
        if video.split('.')[-1]=='MOV':
            image = cv2.rotate(image, cv2.ROTATE_180)
        cv2.imwrite(image_dir+'/%05d.png' % count, image)
        success, image=vidcap.read()
        print(f'Read a new frame {count}: {success}')
        count += 1
    
    print('finish! convert video to frame')

def img_step(img_dir):

    img_list = sorted(glob(os.path.join(img_dir, '*g')))
    output_dir = os.path.join(img_dir, '..', 'step_image')
    os.makedirs(output_dir, exist_ok=True)

    idx = 0
    count = 0

    while count < len(img_list):
        new_name = str(int(idx)).zfill(5) + ".jpg"
        new_path = os.path.join(output_dir, new_name)
        shutil.copy(img_list[count], new_path)
        idx += 1
        count += 9

def slowmo_data(src_dir):
    video2images(src_dir)
    img_dir = os.path.join(src_dir, 'converted_images')
    img_step(img_dir)

    print('slowmo data preprocessing done!')

def visualize_camera_pose():

    # Load the file to understand its structure
    # file_path = './poses_bounds_blender_1.npy'
    file_path = './lego_ref_poses_bounds.npy'
    file_path_1 = './poses_bounds_blender.npy'
    poses_arr = np.load(file_path)
    poses_arr_1 = np.load(file_path_1)

    # Display the shape and a snippet of the data to understand its contents
    # poses_bounds.shape, poses_bounds[:2]  # Show first two entries to understand the structure

    # import pdb; pdb.set_trace()


    # Extracting the camera positions from the first 12 values of each entry
    poses = poses_arr[:, :-2].reshape([-1,3,5]).transpose([1,2,0])  # [3, 5, N]
    # camera_positions = poses_bounds[:, :12].reshape(-1, 3, 4)
    poses_1 = poses_arr_1[:, :-2].reshape([-1,3,5]).transpose([1,2,0])  # [3, 5, N]

    # Extract the translation vectors (camera positions)
    positions = poses[:, 3,:]
    positions_1 = poses_1[:, 3,:]

    mean_pose = np.mean(positions[:, :], axis=1)
    mean_pose_1 = np.mean(positions_1[:, :], axis=1)

    print(np.mean(positions[:, :], axis=1))
    print(np.mean(positions_1[:, :], axis=1))

    positions = positions - np.tile(mean_pose[..., np.newaxis], positions.shape[-1])
    positions_1 = positions_1 - np.tile(mean_pose_1[..., np.newaxis], positions_1.shape[-1])

    # Plotting in 3D
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the camera positions
    ax.scatter(positions[0, :], positions[1, :], positions[2, :], c='r', marker='o', label='Camera colmap Positions')
    ax.scatter(positions_1[0, :], positions_1[1, :], positions_1[2, :], c='b', marker='o', label='Camera blender Positions')

    # Setting labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Visualization of Camera Poses')
    
    # Adding a legend
    ax.legend()

    plt.show()

def viz_cam_pose_dir():

    # Load the file to understand its structure
    file_path = './lego_ref_poses_bounds.npy'
    # file_path = './lego_room_poses_bounds.npy'
    poses_bounds_new = np.load(file_path)

    # Extracting the camera positions from the first 12 values of each entry
    camera_positions_new = poses_bounds_new[:, :12].reshape(-1, 3, 4)

    # Extract the translation vectors (camera positions)
    positions_new = camera_positions_new[:, :, 3]

    # The focal length is the 4th value in each pose row. We'll use this to scale the direction vectors.
    focal_lengths_new = poses_bounds_new[:, 3]

    # The rotation matrices are the first 9 values (3x3 matrix) of each pose
    rotation_matrices_new = camera_positions_new[:, :, :3]

    # The camera direction vector in the camera's local space is typically [0, 0, -1] (pointing along the negative Z-axis)
    camera_directions_new = -rotation_matrices_new[:, :, 2]  # Third column of the rotation matrix

    # Scale direction by focal length (magnitude of the direction vectors)
    direction_vectors_new = camera_directions_new * focal_lengths_new[:, np.newaxis]

    # Plotting in 3D with camera direction vectors
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the camera positions
    ax.scatter(positions_new[:, 0], positions_new[:, 1], positions_new[:, 2], c='r', marker='o', label='Camera Positions')

    # Plot the camera direction vectors
    for i in range(positions_new.shape[0]):
        ax.quiver(positions_new[i, 0], positions_new[i, 1], positions_new[i, 2], 
                direction_vectors_new[i, 0], direction_vectors_new[i, 1], direction_vectors_new[i, 2], 
                length=0.5, color='b')

    # Setting labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Visualization of Camera Poses with Directions (New Data)')

    # Adding a legend
    ax.legend(['Camera Positions', 'Camera Directions'])

    plt.show()
    
def images_slice_rename(img_dir):

    img_list = sorted(glob(os.path.join(img_dir, '*g')))
    output_dir = os.path.join(img_dir, '..', 'step_image')
    os.makedirs(output_dir, exist_ok=True)

    idx = 0
    count = 0

    while count < len(img_list):
        new_name = str(int(idx)).zfill(5) + ".png"
        new_path = os.path.join(output_dir, new_name)
        shutil.copy(img_list[count], new_path)
        idx += 1
        count += 5
        
def preprocess_1(src_dir):
    
    img_dirs = sorted(glob(os.path.join(src_dir, '*/')))
    
    for img_dir in img_dirs:
        print(img_dir)
        img_list = sorted(glob(os.path.join(img_dir, '*G')))
        output_dir = os.path.join(img_dir, 'images')
        os.makedirs(output_dir, exist_ok=True)
        
        for idx, img in enumerate(img_list):
            print(img)
            new_name = str(int(idx)).zfill(5) + ".png"
            new_path = os.path.join(output_dir, new_name)
            shutil.move(img, new_path)
            
def sampling(args):
    
    src_dir = args.src_dir
    step = args.step
    print('src_dir: ', src_dir, 'step: ', step)
    
    img_list = sorted(glob(os.path.join(src_dir, '*g')))
    output_dir = os.path.join(src_dir, '..', 'sampled')
    os.makedirs(output_dir, exist_ok=True)
    
    for idx, img in enumerate(img_list):
               
        if idx % step == 0:
            print(img)
            new_path = os.path.join(output_dir, os.path.basename(img))
            shutil.copy(img, new_path)

def sampling_skip(args):
    
    src_dir = args.src_dir
    step = args.step
    print('src_dir: ', src_dir, 'step: ', step)
    
    img_list = sorted(glob(os.path.join(src_dir, 'images', '*g')))
    msk_list = sorted(glob(os.path.join(src_dir, 'masks', '*g')))
    gt_images_dir = os.path.join(src_dir, 'gt_images')
    gt_masks_dir = os.path.join(src_dir, 'gt_masks')

    os.makedirs(gt_images_dir, exist_ok=True)
    os.makedirs(gt_masks_dir, exist_ok=True)

    for idx in range(len(img_list)):
        if idx % step == 3:
            
            img = img_list[idx]
            msk = msk_list[idx]
            print(img)

            new_img_path = os.path.join(gt_images_dir, os.path.basename(img))
            new_msk_path = os.path.join(gt_masks_dir, os.path.basename(msk))
            shutil.move(img, new_img_path)
            shutil.move(msk, new_msk_path)
            
            
# 이미지 로드 및 밝기 비교 함수
def get_brightness(image):
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 흑백 변환
    return np.mean(gray)  # 평균 밝기 계산
        
# 이미지 대비 및 밝기 조정 함수
def adjust_brightness_contrast(image, alpha=1.0, beta=0):
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

def adjust_brightness(image, beta):
    image = image.astype(np.float32)
    
    adjusted_image = image + beta
    adjusted_image = np.clip(adjusted_image, 0, 255)
    
    return adjusted_image.astype(np.uint8)

def preprocess_2(src_dir):
    image_files = sorted(glob(os.path.join(src_dir, '*g')))
    output_folder = os.path.join(src_dir, '..', 'adjusted_image')
    os.makedirs(output_folder, exist_ok=True)
    
    brightest_image = None
    max_brightness = 0
    for file in image_files:
        image = cv2.imread(file)
        brigthness = get_brightness(image)
        if brigthness > max_brightness:
            max_brightness = brigthness
            brightest_image = image
            
    for file in image_files:
        image = cv2.imread(file)
        brightness = get_brightness(image)
        
        beta = 100
        # alpha = max_brightness / brightness if brightness != 0 else 1.0
        # beta = max_brightness - brightness
        # adjusted_image = adjust_brightness_contrast(image, alpha=1.0, beta=beta)
        adjusted_image = adjust_brightness(image, beta)
        
        output_path = os.path.join(output_folder, os.path.basename(file))
        cv2.imwrite(output_path, adjusted_image)
        
        print(f'Saved adjusted image: {output_path}')
        
def preprocess_kid_running_step_2(src_dir):
    image_files = sorted(glob(os.path.join(src_dir, 'images', '*g')))
    output_folder = os.path.join(src_dir, '..', 'adjusted_images')
    os.makedirs(output_folder, exist_ok=True)
    
    idx = 0
    
    for file in image_files[::3]:
        new_name = f'{idx:05d}.png'
        shutil.copy(file, os.path.join(output_folder, new_name))
        idx += 1
                         
def reverse_mask(src_dir):
    masks_files = sorted(glob(os.path.join(src_dir, 'test_masks', '*g')))
    output_folder = os.path.join(src_dir, 'test_masks_colmap')
    os.makedirs(output_folder, exist_ok=True)
    
    for mask_file in masks_files:
        mask = np.array(Image.open(mask_file).convert("L")) / 255
        mask = 1 - mask
        
        name = mask_file.split('/')[-1]
        output_path = os.path.join(output_folder, name)
        
        cv2.imwrite(output_path, mask * 255)
        
def get_exp_time(src_dir):
    import json
    
    file_path = src_dir
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    import pdb; pdb.set_trace()
    exposures = [frame['exposures'] for frame in data['frames']]
    time_values = [frame['time'] for frame in data['frames']]
    
    output_path = 'exposure_time_train.npy'
    np.save(output_path, {'exposures': exposures, 'time': time_values})
    
def videos2images(args):
    
    src_dir = args.src_dir
    videos_dir = sorted(glob(os.path.join(src_dir, '*.mp4')))
    for video_idx, video_path in enumerate(videos_dir):
        # import pdb; pdb.set_trace()
        images_dir = os.path.join(src_dir, f'cam_{video_idx+1}', 'images')
        os.makedirs(images_dir, exist_ok=True)
        
        vidcap = cv2.VideoCapture(video_path)
        success, image = vidcap.read()
        count = 0
        img_idx = 0
        while success:
            if video_path.split('.')[-1]=='MOV':
                image = cv2.rotate(image, cv2.ROTATE_180)
            if count % 4 == 0: # 120 FPS to 30 FPS
                cv2.imwrite(images_dir+'/%05d.png' % img_idx, image)
                print(f'cam_{video_idx+1} save a new frame {img_idx}')
                img_idx += 1
            success, image=vidcap.read()
            print(f'cam_{video_idx+1} read a new frame {count}: {success}')
            count += 1
    
        print(f'cam_{video_idx + 1} finish! convert video to frame')
    
    # for dir in video_dirs:
    #     dir_path = os.path.join(src_dir, dir)
    #     video2images(dir_path)
        
def sparse_view_processing(args):
    
    src_dir = args.src_dir
    # video_dirs = sorted(glob(os.path.join(src_dir, '*/')))
    start = 20
    N_frame = 36
    step = 4
    end = start + N_frame * step
    new_idx = 0
    output_dir = os.path.join(src_dir, f'i{step}', 'images')
    os.makedirs(output_dir, exist_ok=True)
    
    N_imgs = len(sorted(glob(os.path.join(src_dir, 'cam_1', '*g'))))
    for idx in range(start,end,step):
        if new_idx % 9 == 0:
            image = sorted(glob(os.path.join(src_dir, 'cam_1', '*g')))[idx]
        elif new_idx % 9 == 1:
            image = sorted(glob(os.path.join(src_dir, 'cam_2', '*g')))[idx]
        elif new_idx % 9 == 2:
            image = sorted(glob(os.path.join(src_dir, 'cam_3', '*g')))[idx]
        elif new_idx % 9 == 3:
            image = sorted(glob(os.path.join(src_dir, 'cam_4', '*g')))[idx]
        elif new_idx % 9 == 4:
            image = sorted(glob(os.path.join(src_dir, 'cam_5', '*g')))[idx]
        elif new_idx % 9 == 5:
            image = sorted(glob(os.path.join(src_dir, 'cam_6', '*g')))[idx]
        elif new_idx % 9 == 6:
            image = sorted(glob(os.path.join(src_dir, 'cam_7', '*g')))[idx]
        elif new_idx % 9 == 7:
            image = sorted(glob(os.path.join(src_dir, 'cam_8', '*g')))[idx]
        elif new_idx % 9 == 8:
            image = sorted(glob(os.path.join(src_dir, 'cam_9', '*g')))[idx]
        
        new_name = str(int(new_idx)).zfill(5) + ".png"
        new_path = os.path.join(output_dir, new_name)
        shutil.copy(image, new_path)
        new_idx += 1
        print(image, 'to', new_path)
        
    txt_path = os.path.join(src_dir, f'i{step}', f'119.98fps_({start}, {step}).txt')
        
    with open(txt_path, 'w') as f:
        f.writelines('txt_lines')
    print(f"sparse_view_processing done")
        
    

def multi_view_processing(args):
    
    src_dir = args.src_dir
    # video_dirs = sorted(glob(os.path.join(src_dir, '*/')))
    start = 20
    step = 4
    N_frame = 36
    end = start + step * N_frame
    
    mv_images_temp_dir = os.path.join(src_dir, f'i{step}', 'mv_images_temp')
    mv_images_dir = os.path.join(src_dir, f'i{step}', 'dense', 'mv_images')

    # mv_images_temp_dir = os.path.join(src_dir, 'mv_images_temp')
    # mv_images_dir = os.path.join(src_dir, 'dense', 'mv_images')
    os.makedirs(mv_images_dir, exist_ok=True)

    cams = ['cam_1', 'cam_2', 'cam_3', 'cam_4', 'cam_5', 'cam_6', 'cam_7', 'cam_8', 'cam_9']

    new_idx = 0
    for idx in range(start,end,step):
        output_time_dir = os.path.join(mv_images_temp_dir, f'{new_idx:05d}')
        os.makedirs(output_time_dir, exist_ok=True)
        for cam_idx, cam in enumerate(cams):
            image = sorted(glob(os.path.join(src_dir, cam, '*g')))[idx]
            # new_name = f'cam{cam_idx+1:02d}.png'
            new_name = f'{cam_idx+1:05d}.png'
            new_path = os.path.join(output_time_dir, new_name)
            shutil.copy(image, new_path)
            print(image, 'to', new_path)

        
        # run undistortion
        images_path = multi_view_undistortion(os.path.join(src_dir, f'i{step}'), new_idx)
        # images_path = multi_view_undistortion(src_dir, new_idx)
        
        time_idx_dir = os.path.join(mv_images_dir, f'{new_idx:05d}')
        os.makedirs(time_idx_dir, exist_ok=True)

        for file_path in glob(os.path.join(images_path, '*g')):
            shutil.move(file_path, time_idx_dir)
            print(f"Moved {file_path} -> {time_idx_dir}")

        new_idx = new_idx + 1
        
def multi_view_time_processing(args):
    
    src_dir = args.src_dir
    # video_dirs = sorted(glob(os.path.join(src_dir, '*/')))
    start = args.start
    step = args.step
    N_frame = args.N_frame
    
    ## do not touch ----------------------
    start = start + step // 2
    end = start + step * N_frame
    ### ----------------------------------
    
    mv_images_temp_dir = os.path.join(src_dir, f'i{step}', 'mv_images_temp')
    mv_images_dir = os.path.join(src_dir, f'i{step}', 'dense', 'mv_images_time')

    # mv_images_temp_dir = os.path.join(src_dir, 'mv_images_temp')
    # mv_images_dir = os.path.join(src_dir, 'dense', 'mv_images')
    os.makedirs(mv_images_dir, exist_ok=True)

    cams = ['cam_1', 'cam_2', 'cam_3', 'cam_4', 'cam_5', 'cam_6', 'cam_7', 'cam_8', 'cam_9']

    new_idx = 0
    for idx in range(start,end,step):
        output_time_dir = os.path.join(mv_images_temp_dir, f'{new_idx:05d}')
        os.makedirs(output_time_dir, exist_ok=True)
        for cam_idx, cam in enumerate(cams):
            image = sorted(glob(os.path.join(src_dir, cam, '*g')))[idx]
            # new_name = f'cam{cam_idx+1:02d}.png'
            new_name = f'{cam_idx+1:05d}.png'
            new_path = os.path.join(output_time_dir, new_name)
            shutil.copy(image, new_path)
            print(image, 'to', new_path)

        
        # run undistortion
        images_path = multi_view_undistortion_(mv_images_temp_dir, new_idx)
        # images_path = multi_view_undistortion(src_dir, new_idx)
        
        time_idx_dir = os.path.join(mv_images_dir, f'{new_idx:05d}')
        os.makedirs(time_idx_dir, exist_ok=True)

        for file_path in glob(os.path.join(images_path, '*g')):
            shutil.move(file_path, time_idx_dir)
            print(f"Moved {file_path} -> {time_idx_dir}")

        new_idx = new_idx + 1

def multi_view_processing_crop(args):
    
    src_dir = args.src_dir
    # video_dirs = sorted(glob(os.path.join(src_dir, '*/')))
    step = 4
    start =40
    N_frame = 36
    end = start + step * N_frame



    x_start=203
    x_end=1576
    y_start=94
    y_end=1055
    width=961
    height=1373

    # center = 30
    # start = (center - 8) * step
    # end = (center + 9) * step

    # start = 0 * step
    # end = 72 * step

    # start = 0
    # end = 54
    
    mv_images_temp_dir = os.path.join(src_dir, f'i{step}', 'mv_images_crop')
    mv_images_dir = os.path.join(src_dir, f'i{step}', 'dense', 'mv_images')

    # mv_images_temp_dir = os.path.join(src_dir, 'mv_images_temp')
    # mv_images_dir = os.path.join(src_dir, 'dense', 'mv_images')
    os.makedirs(mv_images_dir, exist_ok=True)

    cams = ['cam_1', 'cam_2', 'cam_3', 'cam_4', 'cam_5', 'cam_6', 'cam_7', 'cam_8', 'cam_9']

    new_idx = 0
    for idx in range(start,end,step):
        # import pdb; pdb.set_trace()
        output_time_dir = os.path.join(mv_images_temp_dir, f'{new_idx:05d}')
        os.makedirs(output_time_dir, exist_ok=True)
        for cam_idx, cam in enumerate(cams):
            image = sorted(glob(os.path.join(src_dir, cam, '*g')))[idx]
            # new_name = f'cam{cam_idx+1:02d}.png'
            new_name = f'{cam_idx+1:05d}.png'
            new_path = os.path.join(output_time_dir, new_name)

            with Image.open(image) as img:
                import pdb; pdb.set_trace()
                # (left, upper, right, lower) 형식으로 crop할 영역을 지정
                # 예시: 왼쪽 위 (0,0)에서 (width, height)까지 자른다고 가정
                cropped = img.crop((x_start, y_start, width, height))
                cropped.save(new_path)


            # shutil.copy(image, new_path)
            print(image, 'to', new_path)

        
        # # run undistortion
        # images_path = multi_view_undistortion(os.path.join(src_dir, f'i{step}'), new_idx)
        # # images_path = multi_view_undistortion(src_dir, new_idx)
        
        # time_idx_dir = os.path.join(mv_images_dir, f'{new_idx:05d}')
        # os.makedirs(time_idx_dir, exist_ok=True)

        # for file_path in glob(os.path.join(images_path, '*g')):
        #     shutil.move(file_path, time_idx_dir)
        #     print(f"Moved {file_path} -> {time_idx_dir}")

        new_idx = new_idx + 1
        

def multi_view_undistortion(dataset_path, time_idx):
    import subprocess

    # Construct the command as a list of arguments

    cmd = [
        "colmap", "image_undistorter",
        "--image_path", os.path.join(dataset_path, "mv_images_temp", f"{time_idx:05d}"),
        "--input_path", os.path.join(dataset_path, "sparse", "0"),
        "--output_path", os.path.join(dataset_path, "mv_images_temp", f"{time_idx:05d}"),
        "--output_type", "COLMAP",
        "--max_image_size", "2000"
    ]

    try:
        subprocess.run(cmd, check=True)
        print("COLMAP undistortion completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Undistortion failed: {e}")

    output_images_path = os.path.join(dataset_path, "mv_images_temp", f"{time_idx:05d}", 'images')

    return output_images_path

def multi_view_undistortion_(dataset_path, time_idx):
    import subprocess

    # Construct the command as a list of arguments

    cmd = [
        "colmap", "image_undistorter",
        "--image_path", os.path.join(dataset_path, f"{time_idx:05d}"),
        "--input_path", os.path.join(dataset_path, "..", "sparse", "0"),
        "--output_path", os.path.join(dataset_path, f"{time_idx:05d}"),
        "--output_type", "COLMAP",
        "--max_image_size", "2000"
    ]

    try:
        subprocess.run(cmd, check=True)
        print("COLMAP undistortion completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Undistortion failed: {e}")

    output_images_path = os.path.join(dataset_path, f"{time_idx:05d}", 'images')

    return output_images_path

def resync_folder_(folder_path, offset):
    # 모든 이미지 파일을 정렬합니다.
    images = sorted(glob(os.path.join(folder_path, '*.png')))
    print(f"[{folder_path}] 원본 파일 개수: {len(images)}")
    
    # offset 만큼의 이미지 삭제
    for img in images[:offset]:
        print(f"삭제: {img}")
        os.remove(img)
    
    # 삭제 후 다시 정렬
    images = sorted(glob(os.path.join(folder_path, '*.png')))
    print(f"[{folder_path}] 삭제 후 파일 개수: {len(images)}")
    
    # 새로운 번호로 재정렬 (00000.png부터)
    for new_idx, img in enumerate(images):
        new_name = os.path.join(folder_path, f"{new_idx:05d}.png")
        print(f"이름 변경: {img} -> {new_name}")
        os.rename(img, new_name)

def resync_folder(args):
    # #Sample 00000
    # for cam in ['cam_1', 'cam_3', 'cam_5', 'cam_7', 'cam_9']:
    #     resync_folder_(os.path.join(args.src_dir, cam), offset=2)

    # # 만약 offset이 1인 카메라(예: cam_8)가 있다면:
    # for cam in ['cam_8']:
    #     resync_folder_(os.path.join(args.src_dir, cam), offset=1)

    # Sample 00001
    for cam in ['cam_1', 'cam_3', 'cam_5', 'cam_7', 'cam_8', 'cam_9']:
        resync_folder_(os.path.join(args.src_dir, cam), offset=2)

    # 만약 offset이 1인 카메라(예: cam_8)가 있다면:
    for cam in ['cam_2', 'cam_4']:
        resync_folder_(os.path.join(args.src_dir, cam), offset=1)




def sparse_view_processing_00000(args):
    
    src_dir = args.src_dir
    # video_dirs = sorted(glob(os.path.join(src_dir, '*/')))

    start = 40
    N_frame = 36
    step = 4
    end = start + N_frame * step
    new_idx = 0
    output_dir = os.path.join(src_dir, f'i{step}', 'images')
    os.makedirs(output_dir, exist_ok=True)

    for idx in range(start,end,step):
        if new_idx % 9 == 0:
            image = sorted(glob(os.path.join(src_dir, 'cam_1', '*g')))[idx+2]
        elif new_idx % 9 == 1:
            image = sorted(glob(os.path.join(src_dir, 'cam_2', '*g')))[idx]
        elif new_idx % 9 == 2:
            image = sorted(glob(os.path.join(src_dir, 'cam_3', '*g')))[idx+2]
        elif new_idx % 9 == 3:
            image = sorted(glob(os.path.join(src_dir, 'cam_4', '*g')))[idx]
        elif new_idx % 9 == 4:
            image = sorted(glob(os.path.join(src_dir, 'cam_5', '*g')))[idx+2]
        elif new_idx % 9 == 5:
            image = sorted(glob(os.path.join(src_dir, 'cam_6', '*g')))[idx]
        elif new_idx % 9 == 6:
            image = sorted(glob(os.path.join(src_dir, 'cam_7', '*g')))[idx+2]
        elif new_idx % 9 == 7:
            image = sorted(glob(os.path.join(src_dir, 'cam_8', '*g')))[idx+1]
        elif new_idx % 9 == 8:
            image = sorted(glob(os.path.join(src_dir, 'cam_9', '*g')))[idx+2]
        
        new_name = str(int(new_idx)).zfill(5) + ".png"
        new_path = os.path.join(output_dir, new_name)
        shutil.copy(image, new_path)
        new_idx += 1
        print(image, 'to', new_path)

from moviepy import VideoFileClip

def mov2mp4(args):
    src_dir = args.src_dir
    videos_path = sorted(glob(os.path.join(src_dir, '*.mov')))
    
    for video_file in videos_path:
        output_file = os.path.splitext(video_file)[0] + ".mp4"
        
        try:
            
            clip = VideoFileClip(video_file)
            clip.write_videofile(output_file, codec="libx264", audio_codec="aac")
            os.remove(video_file)
            print(f"Conversion completed: {output_file}")
        except:
            print(f"An error occurred during conversion")

def gopro(args):
    mov2mp4(args)
    videos2images(args)

def gopro_undistortion(args):

    inv_masks_path = os.path.join(args.src_dir, 'masks')

    inv_masks_files = sorted(glob(os.path.join(inv_masks_path, '*g')))

    output_folder = os.path.join(args.src_dir, 'motion_masks')
    os.makedirs(output_folder, exist_ok=True)
    
    for mask_file in inv_masks_files:
        mask = np.array(Image.open(mask_file).convert("L")) / 255
        mask = 1 - mask

        mask_uint8 = (mask * 255).astype(np.uint8)
        pil_mask = Image.fromarray(mask_uint8, mode='L')
        name = os.path.basename(mask_file).split('.')[0] + ".png"
        output_path = os.path.join(output_folder, name)
        pil_mask.save(output_path, format='PNG')
        
        # name = mask_file.split('/')[-1]
        
        # output_path = os.path.join(output_folder, name)
        # cv2.imwrite(output_path, mask * 255)

    import subprocess


    # Construct the command as a list of arguments

    cmd = [
        "colmap", "image_undistorter",
        "--image_path", os.path.join(args.src_dir, "motion_masks"),
        "--input_path", os.path.join(args.src_dir, "sparse", "0"),
        "--output_path", os.path.join(args.src_dir, "mask-dense"),
        "--output_type", "COLMAP",
        # "--max_image_size", "2000"
    ]

    try:
        subprocess.run(cmd, check=True)
        print("COLMAP undistortion completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Undistortion failed: {e}")

    shutil.move(os.path.join(args.src_dir, "mask-dense", "images/"), os.path.join(args.src_dir, 'dense', 'motion_masks/'))
    convert_gray(os.path.join(args.src_dir, 'dense', 'motion_masks'))

    # output_images_path = os.path.join(dataset_path, "mv_images_temp", f"{time_idx:05d}", 'images')
    return 

def gopro_undistortion_crop(args):

    inv_masks_path = os.path.join(args.src_dir, 'masks')

    inv_masks_files = sorted(glob(os.path.join(inv_masks_path, '*g')))

    output_folder = os.path.join(args.src_dir, 'motion_masks')
    os.makedirs(output_folder, exist_ok=True)
    
    for mask_file in inv_masks_files:
        mask = np.array(Image.open(mask_file).convert("L")) / 255
        mask = 1 - mask

        mask_uint8 = (mask * 255).astype(np.uint8)
        pil_mask = Image.fromarray(mask_uint8, mode='L')
        name = os.path.basename(mask_file).split('.')[0] + ".png"
        output_path = os.path.join(output_folder, name)
        pil_mask.save(output_path, format='PNG')
        
        # name = mask_file.split('/')[-1]
        
        # output_path = os.path.join(output_folder, name)
        # cv2.imwrite(output_path, mask * 255)

    import subprocess

    # Construct the command as a list of arguments

    cmd = [
        "colmap", "image_undistorter",
        "--image_path", os.path.join(args.src_dir, "motion_masks"),
        "--input_path", os.path.join(args.src_dir, "sparse", "0"),
        "--output_path", os.path.join(args.src_dir, "mask-dense"),
        "--output_type", "COLMAP",
        "--max_image_size", "2000"
    ]

    try:
        subprocess.run(cmd, check=True)
        print("COLMAP undistortion completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Undistortion failed: {e}")

    shutil.move(os.path.join(args.src_dir, "mask-dense", "images/"), os.path.join(args.src_dir, 'dense', 'motion_masks/'))
    
    convert_gray(os.path.join(args.src_dir, 'dense', 'motion_masks'))

    # output_images_path = os.path.join(dataset_path, "mv_images_temp", f"{time_idx:05d}", 'images')
    return 

def gopro_gt_undistortion_(args):

    inv_masks_path = os.path.join(args.src_dir, 'gt_masks')

    inv_masks_files = sorted(glob(os.path.join(inv_masks_path, '*g')))

    output_folder = os.path.join(args.src_dir, 'gt_masks_')
    os.makedirs(output_folder, exist_ok=True)
    count=0
    
    for idx, mask_file in enumerate(inv_masks_files):
        mask = np.array(Image.open(mask_file).convert("L")) / 255
        mask = 1 - mask

        mask_uint8 = (mask * 255).astype(np.uint8)
        pil_mask = Image.fromarray(mask_uint8, mode='L')
    
        # name = os.path.basename(mask_file).split('.')[0] + ".png"
        if count in [3, 7, 11, 15, 19, 23, 27, 31, 35]:
            count += 1
            name = f'{count:05d}.png'
        else:
            name = f'{count:05d}.png'

        

        output_path = os.path.join(output_folder, name)

        pil_mask.save(output_path, format='PNG')
        # cv2.imwrite(output_path, mask * 255)
        count += 1

    import subprocess

    # Construct the command as a list of arguments

    cmd = [
        "colmap", "image_undistorter",
        "--image_path", os.path.join(args.src_dir, "gt_masks_"),
        "--input_path", os.path.join(args.src_dir, "sparse", "0"),
        "--output_path", os.path.join(args.src_dir, "gt-mask-dense"),
        "--output_type", "COLMAP",
        "--max_image_size", "2000"
    ]

    try:
        subprocess.run(cmd, check=True)
        print("COLMAP undistortion completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Undistortion failed: {e}")

    msk_list = sorted(glob(os.path.join(args.src_dir, 'gt-mask-dense', 'images', '*g')))

    idx_list = [3, 7, 11, 15, 19, 23, 27, 31, 35]
    new_dir = os.path.join(args.src_dir, 'time', 'masks')
    os.makedirs(new_dir, exist_ok=True)


    for idx, msk_file in enumerate(msk_list):
        new_name = f'{idx_list[idx]:05d}.png'
        new_path = os.path.join(new_dir, new_name)
        shutil.copy(msk_file, new_path)
        
    
    convert_gray(os.path.join(args.src_dir, 'dense', 'motion_masks'))


    # for idx in range(msk_list)

    # shutil.move(os.path.join(args.src_dir, "gt-mask-dense", "images/"), os.path.join(args.src_dir, 'dense', 'motion_masks/'))

    # output_images_path = os.path.join(dataset_path, "mv_images_temp", f"{time_idx:05d}", 'images')

    return 

def gopro_gt_undistortion(args):

    img_path = os.path.join(args.src_dir, 'gt_images')

    img_files = sorted(glob(os.path.join(img_path, '*g')))

    output_folder = os.path.join(args.src_dir, 'gt_images_')
    os.makedirs(output_folder, exist_ok=True)
    count=0
    
    for idx, mask_file in enumerate(img_files):
        # mask = np.array(Image.open(mask_file).convert("L")) / 255
        # mask = 1 - mask

        # mask_uint8 = (mask * 255).astype(np.uint8)
        # pil_mask = Image.fromarray(mask_uint8, mode='L')
    
        # name = os.path.basename(mask_file).split('.')[0] + ".png"
        if count in [3, 7, 11, 15, 19, 23, 27, 31, 35]:
            count += 1
            name = f'{count:05d}.png'
        else:
            name = f'{count:05d}.png'

        

        output_path = os.path.join(output_folder, name)
        shutil.copy(mask_file, output_path)

        # pil_mask.save(output_path, format='PNG')
        # cv2.imwrite(output_path, mask * 255)
        count += 1

    import subprocess

    # Construct the command as a list of arguments

    cmd = [
        "colmap", "image_undistorter",
        "--image_path", os.path.join(args.src_dir, "gt_images_"),
        "--input_path", os.path.join(args.src_dir, "sparse", "0"),
        "--output_path", os.path.join(args.src_dir, "gt-images-dense"),
        "--output_type", "COLMAP",
        "--max_image_size", "2000"
    ]

    try:
        subprocess.run(cmd, check=True)
        print("COLMAP undistortion completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Undistortion failed: {e}")

    msk_list = sorted(glob(os.path.join(args.src_dir, 'gt-images-dense', 'images', '*g')))

    idx_list = [3, 7, 11, 15, 19, 23, 27, 31, 35]
    new_dir = os.path.join(args.src_dir, 'time', 'images')
    os.makedirs(new_dir, exist_ok=True)


    for idx, msk_file in enumerate(msk_list):
        new_name = f'{idx_list[idx]:05d}.png'
        new_path = os.path.join(new_dir, new_name)
        shutil.copy(msk_file, new_path)

        


    # for idx in range(msk_list)

    # shutil.move(os.path.join(args.src_dir, "gt-mask-dense", "images/"), os.path.join(args.src_dir, 'dense', 'motion_masks/'))

    # output_images_path = os.path.join(dataset_path, "mv_images_temp", f"{time_idx:05d}", 'images')

    return 

    
def hdr_hexplane_sampling(args):
    src_dir = args.src_dir
    images_path = sorted(glob(os.path.join(src_dir, 'train', '*g')))
    
    output_dir = os.path.join(src_dir, 'train_step_4', 'images')
    os.makedirs(output_dir, exist_ok=True)
    new_idx = 0
    
    for idx in range(0,len(images_path),4):
        img_file = images_path[idx]
        print(img_file)
        
        new_name = str(int(new_idx)).zfill(5) + ".png"
        new_path = os.path.join(output_dir, new_name)
        
        shutil.copy(img_file, new_path)
        new_idx += 1
        
def hdr_hexplane_sampling_mutant(args):
    src_dir = args.src_dir
    images_path = sorted(glob(os.path.join(src_dir, 'train', '*g')))
    
    output_dir_cam_1 = os.path.join(src_dir, 'cam_1', 'images')
    output_dir_cam_2 = os.path.join(src_dir, 'cam_2', 'images')
    output_dir_cam_3 = os.path.join(src_dir, 'cam_3', 'images')
    
    os.makedirs(output_dir_cam_1, exist_ok=True)
    os.makedirs(output_dir_cam_2, exist_ok=True)
    os.makedirs(output_dir_cam_3, exist_ok=True)

    for idx in range(len(images_path)):
        img_file = images_path[idx]
        print(img_file)
        img = Image.open(img_file)  
        # import pdb; pdb.set_trace()
        
        new_idx = idx // 3
        new_name = str(int(new_idx)).zfill(5) + ".png"
        
        if idx % 3 == 0:
            output_dir = output_dir_cam_1
        elif idx % 3 == 1:
            output_dir = output_dir_cam_2
        else:
            output_dir = output_dir_cam_3
            
        new_path = os.path.join(output_dir, new_name)
        if img.mode == "RGBA":
            img = img.convert("RGB")
            
        img.save(new_path)
            
def hdr_hexplane_sampling_mutant_2(args):
    src_dir = args.src_dir
    images_path_train = sorted(glob(os.path.join(src_dir, 'train', '*g')))
    images_path_test = sorted(glob(os.path.join(src_dir, 'test', '*g')))
    
    output_dir_cam_1 = os.path.join(src_dir, 'cam_1', 'full', 'images')
    output_dir_cam_2 = os.path.join(src_dir, 'cam_2', 'full', 'images')
    output_dir_cam_3 = os.path.join(src_dir, 'cam_3', 'full', 'images')
    
    os.makedirs(output_dir_cam_1, exist_ok=True)
    os.makedirs(output_dir_cam_2, exist_ok=True)
    os.makedirs(output_dir_cam_3, exist_ok=True)
    
    train_idx = 0
    test_idx = 0

    for i in range(len(images_path_train) + len(images_path_test)):
        # import pdb; pdb.set_trace()
        
        idx = i // 3
        
        if idx % 10 == 9:
            img_file = images_path_test[test_idx]
            test_idx += 1
        else:
            img_file = images_path_train[train_idx]
            train_idx += 1
        
        print(img_file)
        img = Image.open(img_file)  
        
        new_idx = idx
        new_name = str(int(new_idx)).zfill(5) + ".png"
        
        if i % 3 == 0:
            output_dir = output_dir_cam_1
        elif i % 3 == 1:
            output_dir = output_dir_cam_2
        else:
            output_dir = output_dir_cam_3
            
        new_path = os.path.join(output_dir, new_name)
        
        print(new_path)
        if img.mode == "RGBA":
            img = img.convert("RGB")
            
        img.save(new_path)
        
        
def hdr_hexplane_cam_1_full(args):
    src_dir = args.src_dir
    images_path_train = sorted(glob(os.path.join(src_dir, 'train', '*g')))
    images_path_test = sorted(glob(os.path.join(src_dir, 'test', '*g')))
    
    output_dir_cam_1 = os.path.join(src_dir, 'cam_1_', 'full', 'images')

    
    os.makedirs(output_dir_cam_1, exist_ok=True)
    train_idx = 0
    test_idx = 0

    for idx in range(len(images_path_train) + len(images_path_test)):        
        
        if idx % 5 == 0:
            img_file = images_path_test[test_idx]
            test_idx += 1
        else:
            img_file = images_path_train[train_idx]
            train_idx += 1
        
        new_idx = idx
        new_name = str(int(new_idx)).zfill(5) + ".png"
        output_dir = output_dir_cam_1
        new_path = os.path.join(output_dir, new_name)
        
        print(img_file)
        print(new_path)
        shutil.copy(img_file, new_path)
        # img = Image.open(img_file)
        # if img.mode == "RGBA":
        #     img = img.convert("RGB")
            
        # img.save(new_path)
        
def json_file_modify(args):
    import json
    train_json = os.path.join(args.src_dir, 'transforms_train.json')
    test_json = os.path.join(args.src_dir, 'transforms_test.json')


    # Load the provided JSON files
    with open(train_json, 'r') as f:
        train_data = json.load(f)

    with open(test_json, 'r') as f:
        test_data = json.load(f)

    # Combine frames from both datasets based on 'time'
    combined_frames = {frame['time']: frame for frame in train_data['frames']}
    for frame in test_data['frames']:
        combined_frames[frame['time']] = frame

    # Sort frames by time
    sorted_frames = [combined_frames[time] for time in sorted(combined_frames.keys())]

    # Update file paths sequentially
    for idx, frame in enumerate(sorted_frames):
        frame['file_path'] = f"./train/{idx:05d}.png"

    # Split into train and test based on the new sequence
    train_frames = [frame for idx, frame in enumerate(sorted_frames) if idx % 2 == 0]
    test_frames = [frame for idx, frame in enumerate(sorted_frames) if idx % 2 == 1]

    # Create train and test JSON structures
    updated_train_data = {
        "camera_angle_x": train_data["camera_angle_x"],
        "camera_angle_y": train_data["camera_angle_y"],
        "near": train_data["near"],
        "far": train_data["far"],
        "frames": train_frames,
    }

    updated_test_data = {
        "camera_angle_x": train_data["camera_angle_x"],
        "camera_angle_y": train_data["camera_angle_y"],
        "near": train_data["near"],
        "far": train_data["far"],
        "frames": test_frames,
    }

    # Save the updated train and test data
    os.makedirs(os.path.join(args.src_dir, 'new_json'))
    updated_train_path = os.path.join(args.src_dir, 'new_json', 'transforms_train.json')
    updated_test_path = os.path.join(args.src_dir, 'new_json', 'transforms_test.json')
    # updated_train_path = '/mnt/data/transforms_train_final.json'
    # updated_test_path = '/mnt/data/transforms_test_final.json'

    with open(updated_train_path, 'w') as f:
        json.dump(updated_train_data, f, indent=4)

    with open(updated_test_path, 'w') as f:
        json.dump(updated_test_data, f, indent=4)

    updated_train_path, updated_test_path
    
def convert(args):
    
    src_dir = args.src_dir
    images_path = sorted(glob(os.path.join(src_dir, '*g')))

    for idx in range(len(images_path)): 
        print(idx)
        img_file = images_path[idx]
        img = Image.open(img_file)
        if img.mode == "RGBA":
            print(img_file)
            img = img.convert("RGB")
        img.save(img_file)

def convert_gray(src_dir):
    
    mask_list = sorted(glob(os.path.join(src_dir, '*g')))

    for idx, mask_file in enumerate(mask_list):
        mask = np.array(Image.open(mask_file).convert("L"))
        pil_mask = Image.fromarray(mask, mode='L')       
        pil_mask.save(mask_file, format='PNG')

    # for idx in range(len(images_path)): 
    #     print(idx)
    #     img_file = images_path[idx]
    #     img = Image.open(img_file)
    #     if img.mode == "RGBA":
    #         print(img_file)
    #         img = img.convert("RGB")
    #     img.save(img_file)

def pixel_viz(args):
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    # 표시할 이미지 경로 설정
    image_path = args.src_img

    # 이미지 읽기
    img = mpimg.imread(image_path)

    # figure와 axis 설정
    fig, ax = plt.subplots()

    # 이미지 표시
    ax.imshow(img)

    # 마우스 커서가 이미지 위에 있을 때 출력할 포맷 설정
    def format_coord(x, y):
        col = int(x + 0.5)
        row = int(y + 0.5)
        
        # 픽셀 범위를 벗어나지 않는지 확인
        if (0 <= row < img.shape[0]) and (0 <= col < img.shape[1]):
            pixel_value = img[row, col]
            return f"x={x:.1f}, y={y:.1f}, pixel={pixel_value}"
        else:
            return f"x={x:.1f}, y={y:.1f}"

    # 축에 마우스 위치에 대한 포맷 지정
    ax.format_coord = format_coord

    plt.title("마우스 커서를 이용해 픽셀 좌표 확인")
    plt.show()

def center_crop_2(args):

    crop_width = 1440
    crop_height = 900

    img_list = sorted(glob(os.path.join(args.src_dir, 'images', '*g')))
    output_dir = os.path.join(args.src_dir, 'cropped', 'images')
    os.makedirs(output_dir, exist_ok=True) 

    for img_path in img_list:
    
        """이미지를 중앙 기준으로 지정된 크기로 crop 하는 함수"""
        with Image.open(img_path) as img:
            width, height = img.size
            
            # 중앙 기준 좌표 계산
            left = (width - crop_width) / 2
            top = (height - crop_height) / 2
            right = (width + crop_width) / 2
            bottom = (height + crop_height) / 2
            
            # 잘라내기
            cropped_img = img.crop((left, top, right, bottom))
            output_path = os.path.join(output_dir, os.path.basename(img_path))
            cropped_img.save(output_path)

    msk_list = sorted(glob(os.path.join(args.src_dir, 'masks', '*g')))
    output_dir = os.path.join(args.src_dir, 'cropped', 'masks')
    os.makedirs(output_dir, exist_ok=True) 

    for msk_path in msk_list:
    
        """이미지를 중앙 기준으로 지정된 크기로 crop 하는 함수"""
        with Image.open(msk_path) as msk:
            width, height = msk.size
            
            # 중앙 기준 좌표 계산
            left = (width - crop_width) / 2
            top = (height - crop_height) / 2
            right = (width + crop_width) / 2
            bottom = (height + crop_height) / 2
            
            # 잘라내기
            cropped_msk = msk.crop((left, top, right, bottom))
            output_path = os.path.join(output_dir, os.path.basename(msk_path))
            cropped_msk.save(output_path)
            print(msk_path)

def sat_analysis(args):
    print("sat_analysis")


    # def normalize_and_mask(image_path):
    #     # 이미지 로드 (그레이스케일)
    #     img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
    #     # [0,1] 정규화
    #     normalized_img = img.astype(np.float32) / 255.0
        
    #     # 마스크 생성 (0.15 이하 또는 0.9 이상)
    #     mask = (normalized_img <= 0.15) | (normalized_img >= 0.9)
        
    #     return normalized_img, mask
    
    # 테스트 예제
    image_path = args.src_img

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        # [0,1] 정규화
    normalized_img = img.astype(np.float32) / 255.0
    
    # 마스크 생성 (0.15 이하 또는 0.9 이상)
    mask = (normalized_img <= 0.15) | (normalized_img >= 0.9)

    # normalized_img, mask = normalize_and_mask(image_path)

    # 마스크 시각화 (0 또는 255로 변환)
    mask_vis = (mask * 255).astype(np.uint8)

            # 마스크 표시
    cv2.imshow("Mask Visualization", mask_vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # cv2.imwrite("mask_output.png", mask_vis)
    
def crop_images(args):
        
    crop_x, crop_y = 50, 0
    crop_w, crop_h = 1600, 2000
    
    input_folder = args.src_dir
    output_folder = os.path.join(input_folder, '..', 'cropped')
    os.makedirs(output_folder, exist_ok=True)
    
    for filename in sorted(os.listdir(input_folder)):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(input_folder, filename)
            img = cv2.imread(img_path)

            # 동일 위치 crop
            cropped = img[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
            
            # 저장
            out_path = os.path.join(output_folder, filename)
            cv2.imwrite(out_path, cropped)
            print(out_path)

    print("Done cropping.")

def crop_masks(args):
    
    """
    args.src_dir = .../dense/cropped/motion_masks
    """
    
    motion_mask_path = args.src_dir 
    motion_mask_files = sorted(glob(os.path.join(motion_mask_path, '*g')))
    
    output_folder = os.path.join(args.src_dir,'..', 'masks')
    os.makedirs(output_folder, exist_ok=True)
    
    for mask_file in motion_mask_files:
        mask = np.array(Image.open(mask_file).convert("L")) / 255
        inv_mask = 1 - mask

        inv_mask_uint8 = (inv_mask * 255).astype(np.uint8)
        pil_inv_mask = Image.fromarray(inv_mask_uint8, mode='L')
        name = os.path.basename(mask_file).split('.')[0] + ".png.png"
        output_path = os.path.join(output_folder, name)
        pil_inv_mask.save(output_path, format='PNG')
        
        mask = np.array(Image.open(mask_file).convert("L"))
        pil_mask = Image.fromarray(mask, mode='L')       
        pil_mask.save(mask_file, format='PNG')


def dslr_preprocess(args):
    print("dslr preprocessing")
    # 입력, 출력 경로 설정
    src_dir = Path(args.src_dir)
    dst_dir = src_dir / "images"
    JPG_dir = src_dir / "JPG"

    # 출력 폴더 없으면 생성
    os.makedirs(dst_dir, exist_ok=True)

    # JPG 파일 목록 정렬
    files = sorted([f for f in os.listdir(JPG_dir) if f.lower().endswith(".jpg")])

    # 리사이즈 크기
    target_size = (1600, 1200)  # (width, height)

    for idx, filename in enumerate(files):
        src_path = JPG_dir / filename
        dst_path = dst_dir / f"{idx:05d}.png"
        
        # 이미지 읽기
        img = cv2.imread(str(src_path))
        if img is None:
            print(f"이미지 로드 실패: {src_path}")
            continue
        
        # 리사이즈
        resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
        
        # 저장
        cv2.imwrite(str(dst_path), resized)

        print(f"Saved {dst_path}")

def mask_undistortion(args):

    inv_masks_path = os.path.join(args.src_dir, 'masks')

    inv_masks_files = sorted(glob(os.path.join(inv_masks_path, '*g')))

    output_folder = os.path.join(args.src_dir, 'motion_masks')
    os.makedirs(output_folder, exist_ok=True)
    
    for mask_file in inv_masks_files:
        mask = np.array(Image.open(mask_file).convert("L")) / 255
        mask = 1 - mask

        mask_uint8 = (mask * 255).astype(np.uint8)
        pil_mask = Image.fromarray(mask_uint8, mode='L')
        name = os.path.basename(mask_file).split('.')[0] + ".png"
        output_path = os.path.join(output_folder, name)
        pil_mask.save(output_path, format='PNG')
        
        # name = mask_file.split('/')[-1]
        
        # output_path = os.path.join(output_folder, name)
        # cv2.imwrite(output_path, mask * 255)

    import subprocess

    # Construct the command as a list of arguments

    cmd = [
        "colmap", "image_undistorter",
        "--image_path", os.path.join(args.src_dir, "motion_masks"),
        "--input_path", os.path.join(args.src_dir, "sparse", "0"),
        "--output_path", os.path.join(args.src_dir, "mask-dense"),
        "--output_type", "COLMAP",
        "--max_image_size", "2000"
    ]

    try:
        subprocess.run(cmd, check=True)
        print("COLMAP undistortion completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Undistortion failed: {e}")

    shutil.move(os.path.join(args.src_dir, "mask-dense", "images/"), os.path.join(args.src_dir, 'dense', 'motion_masks/'))
    convert_gray(os.path.join(args.src_dir, 'dense', 'motion_masks'))

    # output_images_path = os.path.join(dataset_path, "mv_images_temp", f"{time_idx:05d}", 'images')
    return 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir", type=str, help="image directory")
    parser.add_argument("--center_crop", action='store_true', default=False)
    parser.add_argument("--resize_images", action='store_true', default=False)
    parser.add_argument("--mix_t", action='store_true', default=False)
    parser.add_argument("--video2images", action='store_true', default=False)
    parser.add_argument("--slowmo_data", action='store_true', default=False)
    parser.add_argument("--viz_cam_pose", action='store_true', default=False)
    parser.add_argument("--slice_n_rename", action='store_true', default=False)
    parser.add_argument("--mix_12321", action='store_true', default=False)
    parser.add_argument("--preprocess_1", action='store_true', default=False)
    parser.add_argument("--preprocess_2", action='store_true', default=False)
    parser.add_argument("--preprocess_kid", action='store_true', default=False)
    parser.add_argument("--reverse_mask", action='store_true', default=False)
    parser.add_argument("--get_exp_time", action='store_true', default=False)
    parser.add_argument("--videos2images", action='store_true', default=False)
    parser.add_argument("--sparse_view_processing_00000", action='store_true', default=False)
    parser.add_argument("--sparse_view_processing", action='store_true', default=False)
    parser.add_argument("--multi_view_processing", action='store_true', default=False)
    parser.add_argument("--multi_view_time_processing", action='store_true', default=False)
    parser.add_argument("--mov2mp4", action='store_true', default=False)
    parser.add_argument("--gopro", action='store_true', default=False)
    parser.add_argument("--hdr_hexplane_sampling", action='store_true', default=False)
    parser.add_argument("--hdr_hexplane_sampling_mutant", action='store_true', default=False)
    parser.add_argument("--hdr_hexplane_cam_1_full", action='store_true', default=False)
    parser.add_argument("--json_file_modify", action='store_true', default=False)
    parser.add_argument("--sampling", action='store_true', default=False)
    parser.add_argument("--step", type=int)
    parser.add_argument("--convert", action='store_true', default=False)
    parser.add_argument("--resync_folder", action='store_true', default=False)
    parser.add_argument("--sampling_skip", action='store_true', default=False)
    parser.add_argument("--gopro_undistortion", action='store_true', default=False)
    parser.add_argument("--convert_gray", action='store_true', default=False)
    parser.add_argument("--pixel_viz", action='store_true', default=False)
    parser.add_argument("--src_img", type=str, help='image_path')
    parser.add_argument("--center_crop_2", action='store_true', default=False)
    parser.add_argument("--gopro_gt_undistortion", action='store_true', default=False)
    parser.add_argument("--sat_analysis", action='store_true', default=False)
    parser.add_argument("--multi_view_processing_crop", action='store_true', default=False)
    parser.add_argument("--rename_images", action='store_true', default=False)
    parser.add_argument("--crop_images", action='store_true', default=False)
    parser.add_argument("--crop_masks", action='store_true', default=False)
    parser.add_argument("--dslr_preprocess", action='store_true', default=False)
    parser.add_argument("--mask_undistortion", action='store_true', default=False)
    parser.add_argument("--start", type=int, default=None)
    # parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--N_frame", type=int, default=None)

    
    args = parser.parse_args()
    
    # images_slice_rename('/Users/shindy/Projects/hdr-4d/dataset/blender/hdr-hexplane/lego_room/images')
    if args.center_crop:
        center_crop(args.src_dir)
    elif args.resize_images:
        resize_images(args.src_dir)
    elif args.mix_t:
        mix_t(args.src_dir)
    elif args.video2images:
        video2images(args.src_dir)
    elif args.slowmo_data:
        slowmo_data(args.src_dir)
    elif args.viz_cam_pose:
        visualize_camera_pose()
        viz_cam_pose_dir()
    elif args.slice_n_rename:
        images_slice_rename(args.src_dir)
    elif args.mix_12321:
        mix_12321(args.src_dir)
    elif args.preprocess_1:
        preprocess_1(args.src_dir)
    elif args.preprocess_2:
        preprocess_2(args.src_dir)
    elif args.preprocess_kid:
        preprocess_kid_running_step_2(args.src_dir)
    elif args.reverse_mask:
        reverse_mask(args.src_dir)
    elif args.get_exp_time:
        get_exp_time(args.src_dir)
    elif args.videos2images:
        videos2images(args)
    elif args.sparse_view_processing_00000:
        sparse_view_processing_00000(args)
    elif args.sparse_view_processing:
        sparse_view_processing(args)
    elif args.multi_view_processing:
        multi_view_processing(args)
    elif args.multi_view_time_processing:
        multi_view_time_processing(args)
    elif args.mov2mp4:
        mov2mp4(args)
    elif args.gopro:
        gopro(args)
    elif args.hdr_hexplane_sampling:
        hdr_hexplane_sampling(args)
    elif args.hdr_hexplane_sampling_mutant:
        hdr_hexplane_sampling_mutant_2(args)
    elif args.hdr_hexplane_cam_1_full:
        hdr_hexplane_cam_1_full(args)
    elif args.json_file_modify:
        json_file_modify(args)
    elif args.sampling:
        sampling(args)
        
    elif args.convert:
        convert(args)

    elif args.resync_folder:
        resync_folder(args)

    elif args.sampling_skip:
        sampling_skip(args)
    elif args.gopro_undistortion:
        gopro_undistortion(args)
    elif args.gopro_gt_undistortion:
        gopro_gt_undistortion(args)
    elif args.convert_gray:
        convert_gray(args.src_dir)
    elif args.pixel_viz:
        pixel_viz(args)
    elif args.center_crop_2:
        center_crop_2(args)
    elif args.sat_analysis:
        sat_analysis(args)
    elif args.multi_view_processing_crop:
        multi_view_processing_crop(args)
        
    elif args.rename_images:
        rename_images(args)
        
    elif args.crop_images:
        crop_images(args)
        
    elif args.crop_masks:
        crop_masks(args)
    
    elif args.dslr_preprocess:
        dslr_preprocess(args)
    
    elif args.mask_undistortion:
        mask_undistortion(args)