import os
import json
import numpy as np
from typing import List, Dict, Tuple
from .transforms import pose_array_quat_2_matrix
from .convert import *
from .configurations import get_conf_from_dir_name, Configuration, tcp_as_q
from .search import interpolate_linear, binary_search_closest_two_idx, binary_search_closest, sort_by_timestamp
from .utils import load_json, write_json, load_dict_npy
    
_raw_high_freq_fields = ["force_torque", "tcp", "joint"]

class RH20TScene:
    '''
        RH20T data loader with dynamic loading.
        
        TODO: 
        1. Tactile data loading and preprocessing;
        2. Detecting the outliers and process them.
    '''
    def __init__(self, folder:str, robot_confs:List[Configuration]):
        """RH20T scene data loader.

        Args:
            folder (str): scene folder
            robot_confs (list): the list of all robot configurations to search from.
        """
        self._folder = folder
        self._confs = robot_confs
        self._is_action = False
        self._is_high_freq = False
        
        # the subfolder for data with aligned axis and units
        self._used_aligned_folder = "transformed"
        
        self._update()        
        
    def _update(self):
        '''
            Check the validity of folder, and then clear memory consumed
        '''
        if not os.path.exists(self._folder): raise ValueError(f"{self.folder} does not exist")
        if self._folder[-1] == '\\' or self._folder[-1] == '/': self._folder = self._folder[:-1]
        self._parent_folder = os.path.dirname(self.folder)
        
        if os.path.split(self._parent_folder)[-1] == "action":
            self._is_action = True
            self._parent_folder = os.path.dirname(self._parent_folder)
        
        self._conf:Configuration = get_conf_from_dir_name(self._folder, self._confs)
        
        self._metadata = None
        self._intrinsics = None
        self._extrinsics = None
        
        self._high_freq_data_raw = None
        self._tactile_data = None
        
        self._ft_aligned = None
        self._high_freq_aligned = None
        self._tcp_aligned = None
        self._ft_base_aligned = None
        self._tcp_base_aligned = None
        self._base_aligned_timestamps = None
        self._base_aligned_timestamps_in_serial = None
        self._base_aligned_timestamps_time_serial_pairs:List[Tuple[int, str]] = None
        self._joint_angles_aligned:Dict[str, Dict[int, np.ndarray]] = None
        
        self._low_freq_timestamps = None
        self._high_freq_timestamps = None

        self._calib_path = None
        self._calib_tcp = None
        
        self._update_freq_dependency()
    
    def _update_freq_dependency(self):
        self._raw_value = {
            "force_torque": None,
            "gripper_command": None,
            "gripper_info": None,
            "joint": None,
            "tcp": None,
            "tactile": None,
            "timestamps": None
            # TODO: tactile timestamps
        }
        self._raw_value_cam = {
            "force_torque": None,
            "gripper_command": None,
            "gripper_info": None,
            "joint": None,
            "tcp": None,
            "tactile": None
        }
        self._raw_cam_val_loss = {
            "force_torque": None,
            "gripper_command": None,
            "gripper_info": None,
            "joint": None,
            "tcp": None,
            "tactile": None
        }
            
    def _load_calib(self):
        calib_timestamp = self.metadata["calib"]
        if calib_timestamp == -1: raise NotImplementedError
        self._calib_path = os.path.join(self._parent_folder, "calib", str(calib_timestamp))
        self._intrinsics = load_dict_npy(os.path.join(self._calib_path, "intrinsics.npy"))
        self._extrinsics = load_dict_npy(os.path.join(self._calib_path, "extrinsics.npy"))
        for _k in self._extrinsics: self._extrinsics[_k] = self._extrinsics[_k][0]

    def _load_high_freq_data_raw(self): self._high_freq_data_raw = np.load(os.path.join(self._folder, "high_freq_data", "force_torque_tcp_joint_timestamp.npy"))
    def _load_tactile_data(self): self._tactile_data = np.load(os.path.join(self._folder, "high_freq_data", "tactile.npy"))
    
    def _load_raw_value(self, value_field:str):
        if self.is_high_freq and value_field in _raw_high_freq_fields:
            if value_field == "force_torque": 
                _base_idx = 0
                self._raw_value[value_field] = self.high_freq_data_raw[:, _base_idx: _base_idx + self._conf.robot_ft_field[1] - self._conf.robot_ft_field[0]]
            elif value_field == "tcp": 
                _base_idx = self._conf.robot_ft_field[1] - self._conf.robot_ft_field[0]
                self._raw_value[value_field] = self.high_freq_data_raw[:, _base_idx: _base_idx + self._conf.robot_tcp_field[1] - self._conf.robot_tcp_field[0]]
            elif value_field == "joint": 
                _base_idx = self._conf.robot_ft_field[1] - self._conf.robot_ft_field[0] + self._conf.robot_tcp_field[1] - self._conf.robot_tcp_field[0]
                self._raw_value[value_field] = self.high_freq_data_raw[:, _base_idx: _base_idx + self._conf.robot_joint_field[1] - self._conf.robot_joint_field[0]]
        else:
            _time_value = []
            self._raw_value_cam[value_field] = {}

            for serial in self.low_freq_timestamps:
                _cam_path = os.path.join(self.folder, serial_to_cam_dir(serial))
                self._raw_value_cam[value_field][serial] = []
                for _t in self.low_freq_timestamps[serial]:
                    try:
                        self._raw_value_cam[value_field][serial].append(np.load(os.path.join(_cam_path, value_field, str(_t) + ".npy")))
                    except:
                        if self._raw_cam_val_loss[value_field] is None: self._raw_cam_val_loss[value_field] = {}
                        if serial not in self._raw_cam_val_loss[value_field]: self._raw_cam_val_loss[value_field][serial] = []
                        self._raw_cam_val_loss[value_field][serial].append(_t)
                _time_value.extend([(_t, _v) for (_t, _v) in zip(self.low_freq_timestamps[serial], self._raw_value_cam[value_field][serial])])

            _time_value = sorted(_time_value, key=lambda x: x[0])
            self._raw_value[value_field] = [tv[1] for tv in _time_value]
            if self._raw_value["timestamps"] is None: self._raw_value["timestamps"] = [tv[0] for tv in _time_value]
        if value_field == "tactile" and self._conf.tactile: self._raw_value["tactile"] = self.tactile_data[:, :-1]
    
    def _load_ft_aligned(self): 
        self._ft_aligned = load_dict_npy(os.path.join(self.folder, self._used_aligned_folder, "force_torque.npy"))
        sort_by_timestamp(self._ft_aligned)
        for _k in self._ft_aligned: self._ft_aligned[_k] = sorted(self._ft_aligned[_k], key=lambda item:item["timestamp"])
    def _load_high_freq_aligned(self): 
        self._high_freq_aligned = load_dict_npy(os.path.join(self.folder, self._used_aligned_folder, "high_freq_data.npy"))
        sort_by_timestamp(self._high_freq_aligned)
    def _load_tcp_aligned(self): 
        self._tcp_aligned = load_dict_npy(os.path.join(self.folder, self._used_aligned_folder, "tcp.npy"))
        sort_by_timestamp(self._tcp_aligned)
    def _load_ft_base_aligned(self): 
        self._ft_base_aligned = load_dict_npy(os.path.join(self.folder, self._used_aligned_folder, "force_torque_base.npy"))
        sort_by_timestamp(self._ft_base_aligned)
        if self._base_aligned_timestamps is None:
            _t_v = []
            for _k in self._ft_base_aligned: _t_v.extend([(_item["timestamp"], _k, _i) for _i, _item in enumerate(self._ft_base_aligned[_k])])
            _t_v.sort()
            self._base_aligned_timestamps = [_item[0] for _item in _t_v]
            self._base_aligned_timestamps_in_serial = [(_item[1], _item[2]) for _item in _t_v]
            
    def _load_tcp_base_aligned(self): 
        self._tcp_base_aligned = load_dict_npy(os.path.join(self.folder, self._used_aligned_folder, "tcp_base.npy"))
        sort_by_timestamp(self._tcp_base_aligned)
        if self._base_aligned_timestamps is None:
            _t_v = []
            for _k in self._tcp_base_aligned: _t_v.extend([(_item["timestamp"], _k, _i) for _i, _item in enumerate(self._tcp_base_aligned[_k])])
            _t_v.sort()
            self._base_aligned_timestamps = [_item[0] for _item in _t_v]
            self._base_aligned_timestamps_in_serial = [(_item[1], _item[2]) for _item in _t_v]
    
    def _load_joint_angles_aligned(self):
        self._joint_angles_aligned = load_dict_npy(os.path.join(self.folder, self._used_aligned_folder, "joint.npy"))
        if self._base_aligned_timestamps is None:
            _t_v = []
            for _k in self._joint_angles_aligned: _t_v.extend([(_t, _k) for _t in self._joint_angles_aligned[_k]])
            _t_v.sort()
            self._base_aligned_timestamps = [_item[0] for _item in  _t_v]
            self._base_aligned_timestamps_time_serial_pairs = _t_v
        
    def _load_low_freq_timestamps(self):
        # in low frequency data, the timestamps in all the folders (i.e. color, depth, joint, etc.) match
        # therefore here loading only the color data timestamps
        cam_directories = [directory for directory in os.listdir(self.folder) if "cam_" in directory]
        self._low_freq_timestamps:Dict[str, List[int]] = {}
        for cam_directory in cam_directories:
            serial = cam_dir_to_serial(cam_directory)
            # In the resized version, a `timestamps.npy` is provided
            if os.path.exists(os.path.join(self.folder, cam_directory, "timestamps.npy")):
                self._low_freq_timestamps[serial] = sorted(load_dict_npy(os.path.join(self.folder, cam_directory, "timestamps.npy"))["color"])                
                continue
            try: self._low_freq_timestamps[serial] = sorted([img_name_to_timestamp(image_name) for image_name in os.listdir(os.path.join(self.folder, cam_directory, "color"))])
            except Exception as e: 
                print(f"Exception {e} occurs when loading timestamps in {serial}, {self.folder}")
                self._low_freq_timestamps[serial] = []
            
    def _load_high_freq_timestamps(self):
        if self._high_freq_aligned is None: self._load_high_freq_aligned()
        self._high_freq_timestamps:List[int] = sorted([item["timestamp"] for item in self._high_freq_aligned["base"]])
        
    def _load_calib_tcp(self):
        calib_timestamp = self.metadata["calib"]
        if calib_timestamp == -1: raise NotImplementedError
        calib_path = os.path.join(self._parent_folder, "calib", str(calib_timestamp))
        self._calib_tcp = np.load(os.path.join(calib_path, 'tcp.npy')) if self._conf.robot == "ur5" else self._conf.tcp_preprocessor(np.load(os.path.join(calib_path, 'tcp.npy')))
        if len(self._calib_tcp) == 6: self._calib_tcp = tcp_as_q(self._calib_tcp)
    
    @property
    def calib_tcp(self):
        """
            Returns the nearest calibration time tcp in 7d quaternion format.
        """
        if self._calib_tcp is None: self._load_calib_tcp()
        return self._calib_tcp

    def update_zero_offset(self, return_samples:bool=False):
        """
            Update offsets of force and torque sensors given temperature drift.
            To update the offsets, we sample several frames at the beginning where
                the robot arm is considered still, and solve mathematically the offsets.
            Note this function is usually used for preprocessing raw values;
                if you are using preprocessed (aligned) data, you can omit it.
        """
        _fts, _tcps = [], []
        for key in self.low_freq_timestamps:
            if len(self.low_freq_timestamps[key]) == 0: continue
            for _ in range(3):
                # sample for 3 frames; the robot arm is considered still in these frames
                try:
                    _tcp = np.load(os.path.join(self.folder, f"cam_{key}", "tcp", f"{self.low_freq_timestamps[key][_]}.npy"))
                    if self._conf.sensor != "none": _ft = np.load(os.path.join(self.folder, f"cam_{key}", "force_torque", f"{self.low_freq_timestamps[key][_]}.npy"))
                    else: _ft = _tcp[self._conf.robot_ft_field[0]:self._conf.robot_ft_field[1]]
                    _tcp = _tcp[self._conf.robot_tcp_field[0]: self._conf.robot_tcp_field[1]]
                    _fts.append(_ft)
                    _tcps.append(_tcp)
                except:
                    print(f"failed to load {self.low_freq_timestamps[key][0]}.npy, {self.folder}, cam_{key}")
                    continue
        if len(_fts) < 3: print(f"warning: not enough values in {self.folder}")
        self._conf.update_offset(_fts, _tcps)

        return (self._conf, _fts, _tcps) if return_samples else (self._conf)

    @property
    def calib_folder(self): 
        """
            Returns the nearest time calibration result folder.
        """
        if self._calib_path is None: self._load_calib()
        return self._calib_path
    
    @property
    def extrinsics_base_aligned(self) -> Dict[str, np.ndarray]:
        """
            The extrinsics 4x4 matrices for each camera related to robot arm base,
                the matrices are aligned across different robot configurations.
        """
        extrinsics_marker = self.extrinsics
        extrinsics = {}
        base_world_mat = np.linalg.inv(extrinsics_marker[self.in_hand_serials[0]]) @ self._conf.tcp_camera_mat @ np.linalg.inv(pose_array_quat_2_matrix(self.calib_tcp))
        for k in extrinsics_marker: extrinsics[k] = extrinsics_marker[k] @ base_world_mat @ self._conf.align_mat_base
        return extrinsics
        
    ############################## properties ##############################
    @property
    def configuration(self): return self._conf
        
    @property
    def high_freq_data_raw(self):
        """
            Returns raw high frequency data, a 2D numpy.array with
            each row being an array of concatenated force, torque, tcp, 
            joint and timestamp at each timestamp.
        """
        if self._high_freq_data_raw is None: self._load_high_freq_data_raw()
        return self._high_freq_data_raw
    
    @property
    def tactile_data(self):
        """
            Returns raw tactile data.
        """
        if self._tactile_data is None: self._load_tactile_data()
        return self._tactile_data
    
    @property
    def folder(self): return self._folder
    @folder.setter
    def folder(self, val:str):
        self._folder = val
        self._update()
        
    @property
    def is_high_freq(self): return self._is_high_freq
    @is_high_freq.setter
    def is_high_freq(self, val:bool):
        if val and self._conf.sensor == "none": print("Warning: sensor being none, there is no high freq data")
        else: 
            self._is_high_freq = val
            self._update_freq_dependency()
    
    @property
    def used_aligned_folder(self): 
        """
            The subfolder in the scene folder containing aligned data.
        """
        return self._used_aligned_folder
    @used_aligned_folder.setter
    def used_aligned_folder(self, val:str): self._used_aligned_folder = val
    
    @property
    def metadata(self):
        if self._metadata: return self._metadata
        _metadata_path = os.path.join(self.folder, "metadata.json")
        self._metadata = load_json(_metadata_path)
        return self._metadata
    
    @property
    def intrinsics(self) -> Dict[str, np.ndarray]:
        """
            The calibrated intrinsic matrices (3x4) for each camera.
        """
        if self._intrinsics is None: self._load_calib()
        return self._intrinsics
        
    @property
    def extrinsics(self) -> Dict[str, np.ndarray]:
        """
            The calibrated extrinsic matrices (4x4) for each camera.
        """
        if self._extrinsics is None: self._load_calib()
        return self._extrinsics
    
    @property
    def in_hand_serials(self) -> List[str]: return self._conf.in_hand_serial
    
    @property
    def serials(self) -> List[str]:
        """
            Camera serials in this scene data.
        """
        return list(self.low_freq_timestamps.keys())
    
    @property
    def has_tactile(self): return self._conf.tactile
    
    @property
    def conf_num(self): return self._conf.conf_num
        
        
    ############################## frequency dependent properties ##############################
    @property
    def ft_aligned(self):
        if self._ft_aligned is None: self._load_ft_aligned()
        return self._ft_aligned
    
    @property
    def high_freq_aligned(self):
        if self._high_freq_aligned is None: self._load_high_freq_aligned()
        return self._high_freq_aligned
    
    @property
    def tcp_aligned(self):
        if self._tcp_aligned is None: self._load_tcp_aligned()
        return self._tcp_aligned
    
    @property
    def ft_base_aligned(self):
        if self._ft_base_aligned is None: self._load_ft_base_aligned()
        return self._ft_base_aligned
    
    @property
    def tcp_base_aligned(self):
        if self._tcp_base_aligned is None: self._load_tcp_base_aligned()
        return self._tcp_base_aligned
    
    @property
    def joint_angles_aligned(self):
        if self._joint_angles_aligned is None: self._load_joint_angles_aligned()
        return self._joint_angles_aligned
    
    @property
    def low_freq_timestamps(self):
        if self._low_freq_timestamps is None: self._load_low_freq_timestamps()
        return self._low_freq_timestamps
    
    @property
    def high_freq_timestamps(self):
        if self._high_freq_timestamps is None: self._load_high_freq_timestamps()
        return self._high_freq_timestamps
    
    @property
    def start_timestamp_low_freq(self):
        start_timestamp = -1
        for cam_key in self.low_freq_timestamps:
            if len(self.low_freq_timestamps[cam_key]) == 0: continue
            tmp_start_timestamp = self.low_freq_timestamps[cam_key][0]
            if start_timestamp == -1 or start_timestamp > tmp_start_timestamp: start_timestamp = tmp_start_timestamp
        return (None if start_timestamp == -1 else start_timestamp)
    
    @property
    def end_timestamp_low_freq(self):
        end_timestamp = -1
        for cam_key in self.low_freq_timestamps:
            if len(self.low_freq_timestamps[cam_key]) == 0: continue
            tmp_start_timestamp = self.low_freq_timestamps[cam_key][-1]
            if end_timestamp < tmp_start_timestamp: end_timestamp = tmp_start_timestamp
        return (None if end_timestamp == -1 else end_timestamp)
    
    @property
    def start_timestamp_high_freq(self): return self.high_freq_timestamps[0]
    
    @property
    def end_timestamp_high_freq(self): return self.high_freq_timestamps[-1]
    
    
    ############################## frequency independent properties ##############################
    @property
    def start_timestamp(self): return self.start_timestamp_high_freq if self.is_high_freq else self.start_timestamp_low_freq
    
    @property 
    def end_timestamp(self): return self.end_timestamp_high_freq if self.is_high_freq else self.end_timestamp_low_freq
    
    @property
    def start_datetime(self): return timestamp_to_datetime_str(self.start_timestamp)
    
    @property
    def end_datetime(self): return timestamp_to_datetime_str(self.end_timestamp)

    def raw_value_cam(self, value_field:str):
        if self.is_high_freq: raise NotImplementedError("in high frequency mode cannot load cam data")
        if self._raw_value_cam[value_field] is None: self._load_raw_value(value_field)
        return self._raw_value_cam[value_field]

    def raw_cam_val_loss(self, value_field:str):
        if self.is_high_freq: raise NotImplementedError("in high frequency mode cannot load cam data")
        if self._raw_value_cam[value_field] is None: self._load_raw_value(value_field)
        return self._raw_cam_val_loss[value_field]


    def get_audio_path(self): return os.path.join(self._folder, "audio_mixed", os.listdir(os.path.join(self._folder, "audio_mixed"))[0])
        
    ############################## timestamp query methods ##############################
        
    def get_image_path_pairs(self, timestamp:int, image_types:List[str]=["color", "depth"]) -> Dict[str, List[str]]:
        '''
            Get image path pairs given a query timestamp,.
            
            Params:
            ----------
                timestamp:          the query timestamp in milliseconds
                image_types:        the image types retrieved, default to color and depth images
            
            Returns:
            ----------
                image_path_pairs:   dict of image path pairs, keys are serial numbers
        '''
        # images can only be found in low freq data, and vary with cams
        image_path_pairs = {}
        for serial in self.low_freq_timestamps:
            if len(self.low_freq_timestamps[serial]) > 0:
                searched_timestamp = binary_search_closest(self.low_freq_timestamps[serial], timestamp)
                cam_dir, img_name = serial_to_cam_dir(serial), timestamp_to_img_name(searched_timestamp)
                # in the 20T version, color images are compressed into .jpg format to save space
                image_path_pairs[serial] = [os.path.join(self.folder, cam_dir, image_type, img_name) if image_type != "color" else os.path.join(self.folder, cam_dir, image_type, img_name.replace("png", "jpg")) for image_type in image_types]
            else: image_path_pairs[serial] = []
        return image_path_pairs
        
    def get_raw_value(self, timestamp:int, value_field:str) -> np.ndarray:
        """
            Get raw (not aligned) data given a query timestamp and a value field.
            
            Params:
            ----------
                timestamp:          the query timestamp
                value_field:        the query field, supported: force_torque, gripper_command, gripper_info, joint, tcp
            
            Returns:
            ---------
                interpolated_value: the linearly interpolated value
        """
        # force_torque and joint can also be found in high freq data
        # they do not vary with cams
        if self._raw_value[value_field] is None: self._load_raw_value(value_field)
        if self.is_high_freq and value_field in _raw_high_freq_fields:
            _idx_1, _idx_2 = binary_search_closest_two_idx(self.high_freq_timestamps, timestamp)
            return interpolate_linear(
                timestamp, 
                self.high_freq_timestamps[_idx_1], 
                self.high_freq_timestamps[_idx_2], 
                self._raw_value[value_field][_idx_1],
                self._raw_value[value_field][_idx_2]
            )
        _idx_1, _idx_2 = binary_search_closest_two_idx(self._raw_value["timestamps"], timestamp)
        return interpolate_linear(
            timestamp,
            self._raw_value["timestamps"][_idx_1],
            self._raw_value["timestamps"][_idx_2],
            self._raw_value[value_field][_idx_1],
            self._raw_value[value_field][_idx_2]
        )

    # Depreciated, internal use only. Please use get_joint_angles_aligned()
    def get_joints_angles(self, timestamp:int): return self.get_raw_value(timestamp, "joint")[self._conf.robot_joint_field[0]:self._conf.robot_joint_field[1]]
        
    def get_ft_aligned(self, timestamp:int, serial:str="base", zeroed:bool=True) -> np.ndarray:
        zeroed_key = "zeroed" if zeroed else "raw"
        if self.is_high_freq:
            _idx_1, _idx_2 = binary_search_closest_two_idx(self.high_freq_timestamps, timestamp)
            return interpolate_linear(
                timestamp,
                self.high_freq_timestamps[_idx_1],
                self.high_freq_timestamps[_idx_2],
                self.high_freq_aligned[serial][_idx_1][zeroed_key],
                self.high_freq_aligned[serial][_idx_2][zeroed_key]
            )
        if serial == "base":
            ft_base_aligned = self.ft_base_aligned
            _idx_1, _idx_2 = binary_search_closest_two_idx(self._base_aligned_timestamps, timestamp)
            (serial_1, serial_idx_1) = self._base_aligned_timestamps_in_serial[_idx_1]
            (serial_2, serial_idx_2) = self._base_aligned_timestamps_in_serial[_idx_2]
            return interpolate_linear(
                timestamp,
                self._base_aligned_timestamps[_idx_1],
                self._base_aligned_timestamps[_idx_2],
                ft_base_aligned[serial_1][serial_idx_1][zeroed_key],
                ft_base_aligned[serial_2][serial_idx_2][zeroed_key]
            )
        _idx_1, _idx_2 = binary_search_closest_two_idx(self.low_freq_timestamps[serial], timestamp)
        return interpolate_linear(
            timestamp,
            self.low_freq_timestamps[serial][_idx_1],
            self.low_freq_timestamps[serial][_idx_2],
            self.ft_aligned[serial][_idx_1][zeroed_key],
            self.ft_aligned[serial][_idx_2][zeroed_key]
        )
        
    def get_robot_ft_aligned(self, timestamp:int):
        raise NotImplementedError
    
    def get_tcp_aligned(self, timestamp:int, serial:str="base"):
        if self.is_high_freq:
            _idx_1, _idx_2 = binary_search_closest_two_idx(self.high_freq_timestamps, timestamp)
            return interpolate_linear(
                timestamp,
                self.high_freq_timestamps[_idx_1],
                self.high_freq_timestamps[_idx_2],
                self.high_freq_aligned[serial][_idx_1]["tcp"],
                self.high_freq_aligned[serial][_idx_2]["tcp"]
            )
        if serial == "base":
            tcp_base_aligned = self.tcp_base_aligned
            _idx_1, _idx_2 = binary_search_closest_two_idx(self._base_aligned_timestamps, timestamp)
            (serial_1, serial_idx_1) = self._base_aligned_timestamps_in_serial[_idx_1]
            (serial_2, serial_idx_2) = self._base_aligned_timestamps_in_serial[_idx_2]
            return interpolate_linear(
                timestamp,
                self._base_aligned_timestamps[_idx_1],
                self._base_aligned_timestamps[_idx_2],
                tcp_base_aligned[serial_1][serial_idx_1]["tcp"],
                tcp_base_aligned[serial_2][serial_idx_2]["tcp"]
            )
        _idx_1, _idx_2 = binary_search_closest_two_idx(self.low_freq_timestamps[serial], timestamp)
        return interpolate_linear(
            timestamp,
            self.low_freq_timestamps[serial][_idx_1],
            self.low_freq_timestamps[serial][_idx_2],
            self.tcp_aligned[serial][_idx_1]["tcp"],
            self.tcp_aligned[serial][_idx_2]["tcp"]
        )
    
    def get_joint_angles_aligned(self, timestamp:int, serial:str="base"):
        if self.is_high_freq: raise NotImplementedError(f"High freq joint angles getter not implemented yet.")
        if serial == "base":
            joint_angles_aligned = self.joint_angles_aligned
            _idx_1, _idx_2 = binary_search_closest_two_idx(self._base_aligned_timestamps, timestamp)
            (time_1, serial_1) = self._base_aligned_timestamps_time_serial_pairs[_idx_1]
            (time_2, serial_2) = self._base_aligned_timestamps_time_serial_pairs[_idx_2]
            return interpolate_linear(
                timestamp,
                self._base_aligned_timestamps[_idx_1],
                self._base_aligned_timestamps[_idx_2],
                joint_angles_aligned[serial_1][time_1],
                joint_angles_aligned[serial_2][time_2]
            )[self._conf.robot_joint_field[0]:self._conf.robot_joint_field[1]]
        _idx_1, _idx_2 = binary_search_closest_two_idx(self.low_freq_timestamps[serial], timestamp)
        return interpolate_linear(
            timestamp,
            self.low_freq_timestamps[serial][_idx_1],
            self.low_freq_timestamps[serial][_idx_2],
            self.joint_angles_aligned[serial][self.low_freq_timestamps[serial][_idx_1]],
            self.joint_angles_aligned[serial][self.low_freq_timestamps[serial][_idx_2]]
        )[self._conf.robot_joint_field[0]:self._conf.robot_joint_field[1]]
    
    def get_tactile(self, timestamp): raise NotImplementedError
    def detect_outlier(self): raise NotImplementedError
    
    ############################## time interval query ##############################
    
    def get_raw_value_period(self, time_interval:int, value_field:str, start_timestamp:int=None, end_timestamp:int=None):
        """
            Get raw (not aligned) data given a query time interval and a value field.
            
            Params:
            ----------
                time_interval:          the time interval in milliseconds
                value_field:            the query field, supported: force_torque, gripper_command, gripper_info, joint, tcp
                start_timestamp:        the starting timestamp in milliseconds
                end_timestamp:          the ending timestamp in milliseconds
            
            Returns:
            ----------
                raw_value_period:       the linearly interpolated raw values in the interval
        """
        t_start = start_timestamp if start_timestamp else self.start_timestamp
        t_end = end_timestamp if end_timestamp else self.end_timestamp
        raw_value_period = {"val": [], "timestamp": []}
        for _t in range(t_start, t_end, time_interval):
            raw_value_period["val"].append(self.get_raw_value(_t, value_field))
            raw_value_period["timestamp"].append(_t)
        return raw_value_period
    
    def get_image_path_pairs_period(self, time_interval:int, start_timestamp:int=None, end_timestamp:int=None) -> List[Dict[str, List[str]]]:
        '''
            Get [color, depth] image path pairs given time interval.
            
            Params:
            ----------
                time_interval:          the time interval in milliseconds
                start_timestamp:        the starting timestamp in milliseconds
                end_timestamp:          the ending timestamp in milliseconds
            
            Returns:
            ----------
                image_path_pairs_int:   list of dicts of image path pairs, keys are serial numbers and "timestamp"
        '''
        t_start = start_timestamp if start_timestamp else self.start_timestamp
        t_end = end_timestamp if end_timestamp else self.end_timestamp
        image_path_pairs_period = []
        for _t in range(t_start, t_end, time_interval):
            image_path_pairs = self.get_image_path_pairs(_t)
            image_path_pairs["timestamp"] = _t
            image_path_pairs_period.append(image_path_pairs)
        return image_path_pairs_period
    
    
