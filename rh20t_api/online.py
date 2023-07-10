import os
import numpy as np
from .configurations import Configuration, tcp_as_q
from .scene import load_dict_npy
from .transforms import calc_base_world_mat, calc_tcp_camera_mat, calc_tcp_world_mat, matrix_2_pose_array_quat, pose_array_euler_2_pose_array_quat, pose_array_quat_2_matrix

def aligned_tcp_in_base(tcp_base_pose:np.ndarray, conf:Configuration): return matrix_2_pose_array_quat(np.linalg.inv(conf.align_mat_base) @ pose_array_quat_2_matrix(tcp_base_pose) @ conf.align_mat_tcp)
def aligned_tcp_glob_mat(tcp_glob_mat:np.ndarray, conf:Configuration): return tcp_glob_mat @ conf.align_mat_tcp
def aligned_ft_base(ft:np.ndarray, conf:Configuration):
    f_cam = (conf.align_mat_base @ np.array([ft[0], ft[1], ft[2], 0.], dtype=np.float64).reshape(4, 1)).reshape(4,)
    t_cam = (conf.align_mat_base @ np.array([ft[3], ft[4], ft[5], 0.], dtype=np.float64).reshape(4, 1)).reshape(4,)
    return np.array([f_cam[0], f_cam[1], f_cam[2], t_cam[0], t_cam[1], t_cam[2]], dtype=np.float64)

def ft_base_to_cam(base_cam_mat:np.ndarray, ft_base:np.ndarray):
    '''convert force and torque value from base coord to cam coord
    Params:
    ----------
        base_cam_mat:           4x4 base to cam transformation matrix
        ft_base:                force (N) and torque (N m) value concatenated, 6d
    
    Returns:
    ----------
        ft_cam:                 converted force and torque value in cam coord
    '''
    assert base_cam_mat.shape == (4, 4) and ft_base.shape == (6,), f"ft_base_to_cam_err, {base_cam_mat.shape} and {ft_base.shape}"
    f_cam = (base_cam_mat @ np.array([ft_base[0], ft_base[1], ft_base[2], 0.], dtype=np.float64).reshape(4, 1)).reshape(4,)
    t_cam = (base_cam_mat @ np.array([ft_base[3], ft_base[4], ft_base[5], 0.], dtype=np.float64).reshape(4, 1)).reshape(4,)
    return np.array([f_cam[0], f_cam[1], f_cam[2], t_cam[0], t_cam[1], t_cam[2]], dtype=np.float64)

def check_tcp_valid(tcp:np.ndarray):
    if True in np.isnan(tcp): return False
    for _ in tcp:
        if _ != 0: return True
    return False

def raw_force_torque_base(tcp:np.ndarray, force_torque:np.ndarray, conf_info:Configuration):
    '''convert raw force and torque value to base coord
    Params:
    ----------
        tcp:                    tool cartesian pose in base coord, 7d
        force_torque:           raw force (N) and torque (N m) value concatenated, 6d
        conf_info:              configuration information, including zeroing params and tcp dimensions etc.

    Returns:
    ----------
        raw_force_torque:       the zeroed force and torque value in base coord
    '''
    assert tcp.shape == (7,) and force_torque.shape == (6,)
    tcp2sensor_rot = conf_info.tcp2sensor_rot

    sensor_base_mat = pose_array_quat_2_matrix(tcp) @ np.linalg.inv(tcp2sensor_rot)

    force = np.squeeze(sensor_base_mat @ np.asarray([force_torque[0], force_torque[1], force_torque[2], 0.]).reshape(4, 1), axis=(1))
    torque = np.squeeze(sensor_base_mat @ np.asarray([force_torque[3], force_torque[4], force_torque[5], 0.]).reshape(4, 1), axis=(1))

    raw_force_torque = np.concatenate((force[0:3], torque[0:3]), axis=0)
    return raw_force_torque

def zeroed_force_torque_base(tcp:np.ndarray, force_torque:np.ndarray, conf_info:Configuration):
    '''zero the force and torque value in base coord
    Params:
    ----------
        tcp:                    tool cartesian pose in base coord, 7d
        force_torque:           raw force (N) and torque (N m) value concatenated, 6d
        conf_info:              configuration information, including zeroing params and tcp dimensions etc.

    Returns:
    ----------
        zeroed_force_torque:    the zeroed force and torque value in base coord
    '''
    assert tcp.shape == (7,) and force_torque.shape == (6,)
    _offset, _centroid_pos, _gravity, tcp2sensor_rot = conf_info.offset, conf_info.centroid, conf_info.gravity, conf_info.tcp2sensor_rot

    zeroed_force_torque = force_torque - _offset
        
    sensor_base_mat = pose_array_quat_2_matrix(tcp) @ np.linalg.inv(tcp2sensor_rot)
    gravity = np.linalg.inv(sensor_base_mat) @ _gravity
    gravity = gravity[0:3, :].reshape(3,)
    zeroed_force_torque[0:3] -= gravity
    gravity_torque = np.cross(_centroid_pos, gravity[0:3])
    zeroed_force_torque[3:6] -= gravity_torque

    force = np.squeeze(sensor_base_mat @ np.asarray([zeroed_force_torque[0], zeroed_force_torque[1], zeroed_force_torque[2], 0.]).reshape(4, 1), axis=(1))
    torque = np.squeeze(sensor_base_mat @ np.asarray([zeroed_force_torque[3], zeroed_force_torque[4], zeroed_force_torque[5], 0.]).reshape(4, 1), axis=(1))

    zeroed_force_torque = np.concatenate((force[0:3], torque[0:3]), axis=0)
    return zeroed_force_torque

class RH20TOnline:
    def __init__(self, calib_path:str, conf:Configuration, cam_to_project:str=""):
        """
            Online tcp and ft processor for RH20T

            Params:
            ----------
                calib_path:     path to the calibration data
                conf:           the configuration for the online scene
                cam_to_project: the cam coord to be projected to
        """
        self._conf = conf

        self._intrinsics = load_dict_npy(os.path.join(calib_path, "intrinsics.npy"))
        self._extrinsics = load_dict_npy(os.path.join(calib_path, "extrinsics.npy"))
        for _k in self._extrinsics: self._extrinsics[_k] = self._extrinsics[_k][0]
        self._tcp_calib = np.load(os.path.join(calib_path, "tcp.npy"))
        if len(self._tcp_calib) == 6: self._tcp_calib = tcp_as_q(self._tcp_calib)

        self._calc_base_world_mat()

        if cam_to_project not in self._extrinsics or cam_to_project not in self._intrinsics: 
            raise ValueError("Cannot find given serial number in the calib data")
        self._cam_to_project = cam_to_project
    
    def _calc_base_world_mat(self): self._base_world_mat = calc_base_world_mat(self._extrinsics[self._conf.in_hand_serial[0]], self._tcp_calib, self._conf.tcp_camera_mat)
    
    @property
    def cam_to_project(self):
        if self._cam_to_project == "": raise ValueError("The cam serial is not determined yet")
        return self._cam_to_project

    @cam_to_project.setter
    def cam_to_project(self, serial_number:str):
        if serial_number not in self._extrinsics or serial_number not in self._intrinsics: 
            raise ValueError("Cannot find given serial number in the calib data")
        self._cam_to_project = serial_number

    def project_raw_from_robot_sensor(self, raw_data:np.ndarray, is_align:bool=True, is_zero:bool=False):
        """
            Project raw data from the robot sensor saved in [cam_serial]/tcp/[timestamp].npy

            Params:
            ----------
                raw_data:       raw data obtained from the robot sensor, [tcp force torque]
                is_align:       whether to align tcp and coord to unified tcp and coord, defaults to True
                is_zero:        whether to zero force and torque, defaults to False
                                TODO: ONLY IF USING CONFIG 5 YOU SHOULD ENABLE ZERO
            
            Returns:
            ----------
                projected_tcp:  7d np.array
                projected_ft:   6d np.array
        """
        tcp_robot = self._conf.tcp_preprocessor(raw_data[self._conf.robot_tcp_field[0]:self._conf.robot_tcp_field[1]])
        if not check_tcp_valid(tcp_robot): 
            raise ValueError("tcp split from the raw data is invalid, containing np.nan or all being zeros")
        
        ft_robot = raw_data[self._conf.robot_ft_field[0]:self._conf.robot_ft_field[1]]
        if is_zero: ft_robot = zeroed_force_torque_base(tcp_robot, ft_robot, self._conf)
        tcp_base_q = tcp_as_q(tcp_robot) if len(tcp_robot) == 6 else tcp_robot # all tcp records convert to quaternion
        tcp_global_mat = calc_tcp_world_mat(tcp_base_q, base_world_mat=self._base_world_mat)
        if is_align: tcp_global_mat = aligned_tcp_glob_mat(tcp_global_mat, self._conf)
        tcp_camera_mat = calc_tcp_camera_mat(tcp_global_mat, self._extrinsics[self.cam_to_project])
        base_cam_mat = self._extrinsics[self.cam_to_project] @ self._base_world_mat
        if is_align: base_cam_mat = base_cam_mat @ self._conf.align_mat_base
        return matrix_2_pose_array_quat(tcp_camera_mat), ft_base_to_cam(base_cam_mat, ft_robot)

    def project_robottcp_from_cameratcp(self, camera_tcp:np.array):
        """
            Project camera tcp pose to robot base

            Params:
            ----------
                camera_tcp:       camera based tcp predicted by the network, 6d np.array, [xyzrpy]
            
            Returns:
            ----------
                robot_tcp:  7d np.array, [xyz quat]
        """
        camera_tcp = tcp_as_q(camera_tcp)# all tcp records convert to quaternion
        marker_tcp = np.linalg.inv(self._extrinsics[self.cam_to_project]) @ pose_array_quat_2_matrix(camera_tcp)
        robot_tcp = np.linalg.inv(self._base_world_mat) @ marker_tcp
        
        return matrix_2_pose_array_quat(robot_tcp)
        
    def project_raw_from_external_sensor(self, raw_data:np.array, tcp:np.array, is_align=True, is_zero:bool=True):
        """
            Project raw data read from external sensor saved in [cam_serial]/force_torque/[timestamp].npy to a given cam coord
            TODO: PLZ DO NOT USE THIS FUNC IN CONFIG 5

            Params:
            ----------
                raw_data:       raw data obtained from the external sensor, [force torque]
                tcp:            tcp at the same time, simply raw data with same timestamp from the tcp folder is ok
                is_align:       whether align to unified tcp and coord
                is_zero:        whether zero the force and torque
                
            Returns:
            ----------
                projected_ft:   6d np.array
        """
        tcp_robot = self._conf.tcp_preprocessor(tcp[self._conf.robot_tcp_field[0]:self._conf.robot_tcp_field[1]])
        if not check_tcp_valid(tcp_robot): 
            raise ValueError("tcp split from the raw data is invalid, containing np.nan or all being zeros")

        tcp_base_q = tcp_as_q(tcp_robot) if len(tcp_robot) == 6 else tcp_robot # all tcp records convert to quaternion

        base_cam_mat = self._extrinsics[self.cam_to_project] @ self._base_world_mat
        if is_align: base_cam_mat = base_cam_mat @ self._conf.align_mat_base
        ft_base = zeroed_force_torque_base(tcp_base_q, raw_data, self._conf) if is_zero else raw_force_torque_base(tcp_base_q, raw_data, self._conf)
        base_cam_mat = self._extrinsics[self.cam_to_project] @ self._base_world_mat
        if is_align: base_cam_mat = base_cam_mat @ self._conf.align_mat_base
        return ft_base_to_cam(base_cam_mat, ft_base)
        
