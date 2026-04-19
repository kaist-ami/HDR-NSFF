import os
import shutil
import argparse
import glob

def combine_mv_images(src_dir):
    # 원본 이미지들이 들어있는 상위 폴더 경로 (필요에 맞게 수정)
    SRC_DIR = src_dir
    # 이미지를 모아서 저장할 폴더 경로
    DST_DIR = os.path.join(SRC_DIR, "_combined", "images")
    os.makedirs(DST_DIR, exist_ok=True)
    
    folders = sorted([
        f for f in os.listdir(src_dir)
        if os.path.isdir(os.path.join(src_dir, f)) and f.isdigit()
    ])

    # 00000 ~ 00035 폴더를 순회
    for folder_name in folders:
        # folder_name = f"{i:05d}"  # 예: 00000, 00001, ...
        folder_path = os.path.join(SRC_DIR, folder_name)

        # 만약 폴더가 없으면 스킵
        if not os.path.isdir(folder_path):
            continue

        # 각 폴더 내에 00001.png ~ 00009.png 파일을 복사
        for cam_idx in range(1, 10):
            src_filename = f"{cam_idx:05d}.png"   # 예: 00001.png, 00002.png, ...
            src_path = os.path.join(folder_path, src_filename)

            if os.path.isfile(src_path):
                # 이동 후 파일명 예: 00000_1.png, 00000_2.png, ...
                dst_filename = f"{folder_name}_{cam_idx}.png"
                dst_path = os.path.join(DST_DIR, dst_filename)

                # 파일 복사
                shutil.copy2(src_path, dst_path)

    print("Done!")

def rearrange_masks_sequential(src_dir, dst_dir):
    """
    src_dir: 연속된 번호로 된 마스크 파일이 있는 폴더
             예: 00000.png, 00001.png, ... 00323.png
    dst_dir: 결과를 저장할 상위 폴더
             구조 예: mv_masks/00000/00001.png ~ 00009.png, ... mv_masks/00035/00009.png
    """

    # src_dir 내 파일 목록을 정렬하여 가져옴
    files = sorted(f for f in os.listdir(src_dir) if f.lower().endswith(".png"))
    if len(files) == 0:
        print("No PNG files found in:", src_dir)
        return

    # 파일 개수
    num_files = len(files)
    print(f"Found {num_files} mask files in {src_dir}")

    # 파일을 하나씩 순회
    for i, fname in enumerate(files):
        # 예: fname = "00000.png" / i = 0
        #     fname = "00001.png" / i = 1
        #     ...
        # i를 통해 프레임/카메라 인덱스를 계산 (총 324개 가정)
        frame_idx = i // 9        # 0 ~ 35
        camera_idx = (i % 9) + 1  # 1 ~ 9

        # 폴더 이름 (프레임)
        folder_name = f"{frame_idx:05d}"  # 예: 00000 ~ 00035
        # 파일 이름 (카메라)
        mask_name = f"{camera_idx:05d}.png"  # 예: 00001.png ~ 00009.png

        # 실제 경로
        src_path = os.path.join(src_dir, fname)
        dst_folder = os.path.join(dst_dir, folder_name)
        dst_path = os.path.join(dst_folder, mask_name)

        # 폴더 생성
        os.makedirs(dst_folder, exist_ok=True)

        # 파일 복사(또는 이동)
        shutil.copy2(src_path, dst_path)

    print("Mask rearrangement done!")


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir", type=str, help=".../one_arm_swing_shin_1/mv_images")
    parser.add_argument("--mode", type=str, help='combine / rearrange')
    
    args = parser.parse_args()
    
    # src_dir = '/home/shindy/projects/hdr-4d/hdr-nsff/dataset_home/gopro/i8/one_arm_swing_shin_1/mv_images'
    if args.mode == 'combine':
        combine_mv_images(args.src_dir)
    elif args.mode == 'rearrange':
        # It shoule be done after SAM2
        combined_mv_images_path = os.path.join(args.src_dir, '_combined', 'motion_masks')
        mv_masks_path = os.path.join(args.src_dir, '..','mv_masks')
        rearrange_masks_sequential(combined_mv_images_path, mv_masks_path)

    elif args.mode == 'rearrange_time':
        combined_mv_images_path = os.path.join(args.src_dir, '_combined', 'motion_masks')
        mv_masks_path = os.path.join(args.src_dir, '..','mv_masks_time')
        rearrange_masks_sequential(combined_mv_images_path, mv_masks_path)

    else:
        print("You didn't set mode. What do you want? combine or rearrange")