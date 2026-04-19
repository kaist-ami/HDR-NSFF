"""Resize images to a target height while preserving aspect ratio.

Output directory: <data_path>/images_{W}x{H}/
This matches the directory name expected by load_llff._load_data(height=...).
"""
import argparse
import cv2
import os

def resize_images(data_path, resize_height):
    img_dir = os.path.join(data_path, 'images')
    imgs = sorted([f for f in os.listdir(img_dir)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    if not imgs:
        raise FileNotFoundError(f'No images found in {img_dir}')

    h0, w0 = cv2.imread(os.path.join(img_dir, imgs[0])).shape[:2]
    factor = h0 / float(resize_height)
    resize_width = int(round(w0 / factor))

    out_dir = os.path.join(data_path, f'images_{resize_width}x{resize_height}')
    if os.path.exists(out_dir):
        print(f'Already exists, skipping: {out_dir}')
        return

    os.makedirs(out_dir)
    print(f'Resizing {len(imgs)} images: {w0}x{h0} -> {resize_width}x{resize_height}')
    for fname in imgs:
        img = cv2.imread(os.path.join(img_dir, fname))
        resized = cv2.resize(img, (resize_width, resize_height), interpolation=cv2.INTER_AREA)
        out_name = os.path.splitext(fname)[0] + '.png'
        cv2.imwrite(os.path.join(out_dir, out_name), resized)
    print(f'Done: {out_dir}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--resize_height', type=int, default=360)
    args = parser.parse_args()
    resize_images(args.data_path, args.resize_height)
