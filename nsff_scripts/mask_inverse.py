import os
import numpy as np

from glob import glob
from PIL import Image
import argparse

def mask_inverse(mask_path, save_path):
    mask = Image.open(mask_path)
    mask = np.array(mask)
    mask = 255 - mask
    mask = Image.fromarray(mask)
    mask.save(save_path)
    
def run(args):
    
    mask_dir = os.path.join(args.data_path, '..', 'masks_inverse')
    mask_list = sorted(glob(os.path.join(mask_dir, '*g')))
    
    for mask_file in mask_list:
        mask_name = os.path.basename(mask_file)      
        save_dir = os.path.join(args.data_path, 'motion_masks')
        os.makedirs(save_dir, exist_ok=True)
        save_mask_path = os.path.join(save_dir, mask_name)
        mask_inverse(mask_file, save_mask_path)
    
    print('Inverse masks saved in {}'.format(save_dir))
    
def convert_gray(src_dir):
    
    mask_list = sorted(glob(os.path.join(src_dir, '*g')))
    print(f'Number of masks: {len(mask_list)}')

    for idx, mask_file in enumerate(mask_list):
        mask = np.array(Image.open(mask_file).convert("L"))
        pil_mask = Image.fromarray(mask, mode='L')       

        # output_path = os.path.join(output_folder, name)

        pil_mask.save(mask_file, format='PNG')
        
    print("done!")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='../data_path/dense')
    parser.add_argument('--convert_gray', action='store_true', default=False)
    args = parser.parse_args()
    
    if args.convert_gray:
        convert_gray(args.data_path)
    else:
        run(args)