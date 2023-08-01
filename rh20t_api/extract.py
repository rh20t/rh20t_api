import os
import cv2
import time
import numpy as np
from PIL import Image
from multiprocessing import Pool


target_color = '/path/to/color.mp4'
target_depth = '/path/to/depth.mp4'
target_timestamps = '/path/to/timestamps.npy'
dest_dir = '/path/to/destination/folder'
size = (720, 1280)  # or (360, 640) in resized version

if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)

dest_color_dir = os.path.join(dest_dir, 'color')
dest_depth_dir = os.path.join(dest_dir, 'depth')
if not os.path.exists(dest_color_dir):
    os.makedirs(dest_color_dir)
if not os.path.exists(dest_depth_dir):
    os.makedirs(dest_depth_dir)

meta = np.load(target_timestamps, allow_pickle = True).item()

### Color ###
meta_color = meta['color']
cap = cv2.VideoCapture(target_color)
cnt = 0
while True:
    ret, frame = cap.read()
    if ret:
        cv2.imwrite(os.path.join(dest_color_dir, '{}.jpg'.format(meta_color[cnt])), frame)
        cnt += 1
    else:
        break
cap.release()

## Depth ###
meta_depth = meta['depth']
cap = cv2.VideoCapture(target_depth)
cnt = 0
while True:
    ret, frame = cap.read()
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray1 = np.array(gray[:720, :]).astype(np.int32)
        gray2 = np.array(gray[720:, :]).astype(np.int32)
        gray = gray2 * 256 + gray1
        gray = np.array(gray).astype(np.uint16)
        cv2.imwrite(os.path.join(dest_depth_dir, '{}.png'.format(meta_depth[cnt])), gray)
        cnt += 1
    else:
        break
cap.release()
