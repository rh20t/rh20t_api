"""
Scripts and sample usages to convert RH20T to image version.
This script should be executed after unzipped the file if you want to use RH20T_api functions.
"""
import os
import cv2
import time
import numpy as np
from PIL import Image
from multiprocessing import Pool


################################## Convert RH20T to image version ##################################

def convert_color(color_file, color_timestamps, dest_color_dir):
    """
    Args:
    - color_file: the color video file;
    - color_timestamps: the color timestamps;
    - dest_color_dir: the destination color directory.
    """
    cap = cv2.VideoCapture(color_file)
    cnt = 0
    while True:
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(os.path.join(dest_color_dir, '{}.jpg'.format(color_timestamps[cnt])), frame)
            cnt += 1
        else:
            break
    cap.release()

def convert_depth(depth_file, depth_timestamps, dest_depth_dir, size = (1280, 720)):
    """
    Args:
    - depth_file: the depth video file (special encoded);
    - depth_timestamps: the depth timestamps;
    - dest_depth_dir: the destination depth directory;
    - size: the size of the depth map ( (640, 360) for resized version ).
    """
    width, height = size
    cap = cv2.VideoCapture(depth_file)
    cnt = 0
    while True:
        ret, frame = cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray1 = np.array(gray[:height, :]).astype(np.int32)
            gray2 = np.array(gray[height:, :]).astype(np.int32)
            gray = np.array(gray2 * 256 + gray1).astype(np.uint16)
            cv2.imwrite(os.path.join(dest_depth_dir, '{}.png'.format(depth_timestamps[cnt])), gray)
            cnt += 1
        else:
            break
    cap.release()

def convert_dir(color_file, timestamps_file, dest_dir, depth_file = None, size = (1280, 720)):
    """
    Args:
    - color_file: the color video file;
    - timestamps_file: the timestamps file;
    - dest_dir: the destination directory;
    - depth_file: the depth video file (special encoded), set to None if no depth usage;
    - size: the size of the depth map ( (640, 360) for resized version ).
    """
    with_depth = (depth_file is not None)
    assert os.path.exists(color_file)
    assert os.path.exists(timestamps_file)
    if with_depth: 
        # with depth, but no depth map is generated for the scene due to technical issues.
        if not os.path.exists(depth_file):
            with_depth = False
    meta = np.load(timestamps_file, allow_pickle = True).item()
    dest_color_dir = os.path.join(dest_dir, 'color')
    if not os.path.exists(dest_color_dir):
        os.makedirs(dest_color_dir)
    convert_color(color_file = color_file, color_timestamps = meta['color'], dest_color_dir = dest_color_dir)
    if with_depth:
        dest_depth_dir = os.path.join(dest_dir, 'depth')
        if not os.path.exists(dest_depth_dir):
            os.makedirs(dest_depth_dir) 
        convert_depth(depth_file = depth_file, depth_timestamps = meta['depth'], dest_depth_dir = dest_depth_dir, size = size)

def convert_scene(scene_root_dir, dest_scene_dir, scene_depth_dir = None, size = (1280, 720)):
    """
    Args:
    - scene_root_dir: the root directory for the current scene;
    - dest_scene_dir: the destination root directory for the current scene (set to the same as scene_root_dir to extract into the original directory);
    - dest_dir: the destination scene directory;
    - depth_file: the depth video file (special encoded), set to None if no depth usage;
    - size: the size of the depth map ( (640, 360) for resized version ).
    """
    assert os.path.exists(scene_root_dir)
    for cam_folder in os.listdir(scene_root_dir):
        if "cam_" not in cam_folder:
            continue
        convert_dir(
            color_file = os.path.join(scene_root_dir, cam_folder, 'color.mp4'),
            timestamps_file = os.path.join(scene_root_dir, cam_folder, 'timestamps.npy'),
            dest_dir = os.path.join(dest_scene_dir, cam_folder),
            depth_file = None if scene_depth_dir is None else os.path.join(scene_depth_dir, cam_folder, 'depth.mp4'),
            size = size
        )

def convert_rh20t(root_dir, dest_dir, depth_dir = None, num_workers = 20):
    """
    Args:
    - root_dir: the root directory for RH20T;
    - dest_dir: the destination root directory for RH20T in image version (it can be the same as root_dir);
    - depth_dir: the root directory for RH20T depth file;
    - num_workers: the number of workers in multiprocessing.
    """
    assert os.path.exists(root_dir)
    if 'resized' in root_dir:
        size = (640, 360)
    else:
        size = (1280, 720)
    with Pool(processes = num_workers) as pool:
        for cfg_folder in os.listdir(root_dir):
            cfg_root_dir = os.path.join(root_dir, cfg_folder)
            dest_cfg_dir = os.path.join(dest_dir, cfg_folder)
            cfg_depth_dir = (None if depth_dir is None else os.path.join(depth_dir, cfg_folder))
            for scene_folder in os.listdir(cfg_root_dir):
                pool.apply_async(convert_scene, args = (
                    os.path.join(cfg_root_dir, scene_folder),
                    os.path.join(dest_cfg_dir, scene_folder),
                    None if cfg_depth_dir is None else os.path.join(cfg_depth_dir, scene_folder),
                    size
                ))
        pool.close()
        pool.join()
        


################################## Sample Usage ##################################

if __name__ == '__main__':
    # 1. For full version (or ignore depth_dir if no depth usage)
    convert_rh20t(
        root_dir = "/path/to/RH20T/", 
        dest_dir = "/path/to/destination/RH20T/", 
        depth_dir = "/path/to/RH20T_depth/",
        num_workers = 20
    )
    """
    # 2. For resized version.
    convert_rh20t(
        root_dir = "/path/to/RH20T_resized/", 
        dest_dir = "/path/to/destination/RH20T_resized/", 
        depth_dir = "/path/to/RH20T_resized_depth/",
        num_workers = 20
    )
    """