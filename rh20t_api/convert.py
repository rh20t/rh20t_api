"""
    String conversion for file names, time, camera serial etc.
"""

from datetime import datetime

def img_name_to_timestamp(file_name:str): return int(file_name[:13])
def timestamp_to_img_name(timestamp:int): return str(timestamp) + ".png"

def npy_to_timestamp(file_name:str): return int(file_name[:13])
def timestamp_to_npy(timestamp:int): return str(timestamp) + ".npy"

def serial_to_cam_dir(serial:str): return "cam_" + serial
def cam_dir_to_serial(cam_dir:str): return cam_dir.replace("cam_", "")

def timestamp_to_datetime_str(timestamp:int): return datetime.fromtimestamp(timestamp / 1000.0).strftime("%Y-%m-%d %H:%M:%S")