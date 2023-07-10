import json
import numpy as np
from typing import List
from .transforms import pose_array_euler_2_pose_array_quat, pose_array_quat_2_matrix, \
    pose_array_rotvec_2_matrix, matrix_2_pose_array_quat, pose_array_quat_2_pose_array_euler
from transforms3d.quaternions import axangle2quat

def bound_to_0_2pi(num): return (num + np.pi * 2 if num < 0 else num)

def identical_tcp_preprocessor(tcp:np.ndarray): return tcp
def kuka_tcp_preprocessor(tcp:np.ndarray): return np.array([tcp[0] / 1000., tcp[1] / 1000., tcp[2] / 1000., bound_to_0_2pi(tcp[3]), bound_to_0_2pi(tcp[4]), bound_to_0_2pi(tcp[5])], dtype=np.float64)
def franka_tcp_preprocessor(tcp:np.ndarray): return pose_array_quat_2_pose_array_euler(matrix_2_pose_array_quat(pose_array_rotvec_2_matrix(tcp) @ np.array([[1., 0., 0., 0.], [0., -1., 0., 0.], [0., 0., -1., 0], [0., 0., 0., 1.]], dtype=np.float64)))
def ur5_tcp_preprocessor(tcp:np.ndarray): return np.array([tcp[0], tcp[1], tcp[2]] + axangle2quat(tcp[3:6], np.linalg.norm(tcp[3:6])).tolist(), dtype=np.float64)

tcp_preprocessors = {
    "flexiv": identical_tcp_preprocessor,
    "ur5": ur5_tcp_preprocessor,
    "franka": franka_tcp_preprocessor,
    "kuka": kuka_tcp_preprocessor
}

def tcp_as_q(tcp:np.ndarray):
    assert tcp.shape == (6,), tcp
    tcp_q = pose_array_euler_2_pose_array_quat(tcp[0:6])
    tcp_q = np.concatenate((tcp_q, tcp[6:]), axis=0)
    return tcp_q

calib_tcp_preprocessors = {
    "flexiv": identical_tcp_preprocessor,
    "ur5": identical_tcp_preprocessor,
    "franka": franka_tcp_preprocessor,
    "kuka": kuka_tcp_preprocessor
}

def robotiq_gripper_preprocessor(gripper_cmd:np.ndarray): return [(255 - gripper_cmd[0]) / 255 * 85, gripper_cmd[1], gripper_cmd[2]]
def dahuan_gripper_preprocessor(gripper_cmd:np.ndarray): return [gripper_cmd[0] / 1000 * 95, gripper_cmd[1], gripper_cmd[2]]
def wsg_gripper_preprocessor(gripper_cmd:np.ndarray): return [gripper_cmd[0] / 1.1, gripper_cmd[1], gripper_cmd[2]]
def franka_gripper_preprocessor(gripper_cmd:np.ndarray): return [gripper_cmd[0] / 100, gripper_cmd[1], gripper_cmd[2]]

gripper_preprocessors = {
    "Dahuan AG-95": dahuan_gripper_preprocessor, 
    "WSG-50": wsg_gripper_preprocessor, 
    "Robotiq 2F-85": robotiq_gripper_preprocessor, 
    "franka": franka_gripper_preprocessor
}

valid_sensors = ["dahuan", "ati", "none"]
valid_grippers = ["Dahuan AG-95", "WSG-50", "Robotiq 2F-85", "franka"]

class Configuration:
    def __init__(self, conf_dict):
        self.conf_num:int = conf_dict["conf_num"]
        self.robot:str = conf_dict["robot"]
        self.robot_urdf:str = conf_dict["robot_urdf"]
        self.robot_mesh:str = conf_dict["robot_mesh"]
        self.robot_tcp_field:List[int] = conf_dict["robot_tcp_field"]
        self.robot_ft_field:List[int] = conf_dict["robot_ft_field"]
        self.robot_joint_field:List[int] = conf_dict["robot_joint_field"]
        self.robot_joint_sequence:List[str] = conf_dict["robot_joint_sequence"]
        self.sensor:str = conf_dict["sensor"]
        self.sensor_ft_field:List[int] = conf_dict["sensor_ft_field"]
        self.offset:np.ndarray = np.array(conf_dict["offset"], dtype=np.float64)
        self.centroid:np.ndarray = np.array(conf_dict["centroid"], dtype=np.float64)
        self.gravity:np.ndarray = np.array(conf_dict["gravity"], dtype=np.float64).reshape(4, 1)
        self.tcp2sensor_rot:np.ndarray = np.array(conf_dict["ts_rot"], dtype=np.float64)
        self.in_hand_serial:List[str] = conf_dict["in_hand"]
        self.tcp_camera_mat:np.ndarray = np.array(conf_dict["tc_mat"], dtype=np.float64)
        self.align_mat_tcp:np.ndarray = np.array(conf_dict["align_mat_tcp"], dtype=np.float64)
        self.align_mat_base:np.ndarray = np.array(conf_dict["align_mat_base"], dtype=np.float64)
        self.tactile:bool = conf_dict["tactile"]
        self.gripper:str = conf_dict["gripper"]

        self.tcp_preprocessor = tcp_preprocessors[self.robot]
        self.calib_tcp_preprocessor = calib_tcp_preprocessors[self.robot]
        self.gripper_preprocessor = gripper_preprocessors[self.gripper]

        assert len(self.robot_tcp_field) == 2
        assert len(self.robot_ft_field) == 2
        assert len(self.sensor_ft_field) == (0 if self.sensor == "none" else 2)
        assert self.offset.shape == (6,)
        assert self.centroid.shape == (3,)
        assert self.tcp2sensor_rot.shape == (4, 4)
        assert self.tcp_camera_mat.shape == (4, 4)
        assert self.sensor in valid_sensors
        assert self.gripper in valid_grippers
    
    def update_offset(self, raw_fts:np.ndarray, raw_tcps:np.ndarray):
        """
            Considering temperature drift, the force and torque sensor offset 
                should be updated according to the values sampled at the beginning 
                of a scene (when the the robot arm is considered to be still)
            
            Params:
            ----------
                raw_fts:    the sampled still force and torque values
                raw_tcps:   the sampled still tcps
            
            Returns:
            ----------
                None
        """
        # print("updating offset...")
        np.set_printoptions(precision=3, suppress=True)
        # print("original offset:", self.offset)
        _Fg_homos, _offsets = [], []
        for (raw_ft, raw_tcp) in zip(raw_fts, raw_tcps):
            # print("ft:", raw_ft, "tcp:", raw_tcp)
            raw_tcp = self.tcp_preprocessor(raw_tcp)
            _tcp = tcp_as_q(raw_tcp) if len(raw_tcp) == 6 else raw_tcp
            _tcp = pose_array_quat_2_matrix(_tcp)
            _centroid, _gravity = self.centroid, self.gravity[:3].reshape(3,)
            _gravity_torque = np.cross(_centroid, _gravity)
            _force, _torque = raw_ft[0:3], raw_ft[3:6]
            _T_offset = _torque - _gravity_torque
            _Fg_homo = np.array([_gravity[0], _gravity[1], _gravity[2], 0]).reshape(4, 1)
            _Fg_homos.append(_Fg_homo)
            _F_offset = _force - (self.tcp2sensor_rot @ np.linalg.inv(_tcp) @ _Fg_homo).reshape(4,)[:3]
            _offsets.append(np.concatenate([_F_offset, _T_offset]))
        self.offset = np.mean(_offsets, axis=0)
        # print("computed new offsets")
        # for _offset in _offsets: print(_offset)
        # print("new offset:", self.offset)
        
def load_conf(path:str):
    '''load configurations
    Params:
    ----------
        path:   configuration file path
    Returns:
    ----------
        confs:  configurations  
    '''
    _f = open(path, "r")
    _confs = json.load(_f)
    confs = [Configuration(_conf) for _conf in _confs]
    _f.close()
    return confs

def get_conf_from_dir_name(scene_dir_name:str, confs:list):
    '''
        get the Configuration entity given the scene directory name
        
        Params:
        ----------
        scene_dir_name:     the given scene directory name
        confs:              the loaded Configuration entities
        parent_folder:      the parent folder name of the scene directory
        is_action:          if the folder is a scene or action
    '''
    if scene_dir_name[-1] == "/": scene_dir_name = scene_dir_name[:-1]
    scene_dir_name = scene_dir_name.split("/")[-1]
    conf_num = 0
    if 'cfg_0001' in scene_dir_name: conf_num = 1
    elif 'cfg_0002' in scene_dir_name: conf_num = 2
    elif 'cfg_0003' in scene_dir_name: conf_num = 3
    elif 'cfg_0004' in scene_dir_name: conf_num = 4
    elif 'cfg_0005' in scene_dir_name: conf_num = 5
    elif 'cfg_0006' in scene_dir_name: conf_num = 6
    elif 'cfg_0007' in scene_dir_name: conf_num = 7
    else: raise NotImplementedError
    for _c in confs: 
        if _c.conf_num == conf_num: return _c
    return None

def get_conf_by_conf_num(confs:list, conf_num:int):
    if conf_num > len(confs): raise ValueError("conf_num is greater than provided confs")
    return confs[conf_num - 1]