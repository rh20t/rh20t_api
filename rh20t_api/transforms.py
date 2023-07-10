"""
    Functions related to 3D geometric transformations.
    Mainly implements:
    1. transformations between different 3D pose/transformation representations;
    2. relative transformations for robot arm base, world(marker), tcp and fixed camera
    
    Implemented representations: 
    - quaternion, 7d
    - matrix, 4x4
    - axis rotation vector, 6d
    - euler angles, 6d
    
    | pose representation | quaternion | matrix | rotation vector | euler |
    |---------------------|------------|--------|-----------------|-------|
    |     quaternion      |     -      |   √    |        √        |   √   |
    |       matrix        |     √      |   -    |                 |       |
    |   rotation vector   |     √      |   √    |        -        |       |
    |        euler        |     √      |        |                 |   -   |
"""

import numpy as np
from transforms3d.euler import euler2quat, quat2euler
from transforms3d.quaternions import quat2mat, mat2quat, axangle2quat, quat2axangle
import cv2

def pose_array_quat_2_matrix(pose:np.ndarray):
    '''transform pose array of quaternion to transformation matrix

    Param:
        pose:   7d vector, with t(3d) + q(4d)
    ----------
    Return:
        mat:    4x4 matrix, with R,T,0,1 form
    '''
    if pose.shape != (7,): raise ValueError
    mat = quat2mat([pose[3], pose[4], pose[5], pose[6]])
    return np.array([[mat[0][0], mat[0][1], mat[0][2], pose[0]],
                    [mat[1][0], mat[1][1], mat[1][2], pose[1]],
                    [mat[2][0], mat[2][1], mat[2][2], pose[2]],
                    [0,0,0,1]])

def matrix_2_pose_array_quat(mat:np.ndarray):
    '''transform transformation matrix to pose array of quaternion

    Param:
        mat:        4x4 matrix, with R,T,0,1 form
    ----------
    Return:
        pose:       7d vector, with t(3d) + q(4d)
    '''
    if mat.shape != (4, 4): raise ValueError
    rotation_mat = np.array([[mat[0][0], mat[0][1], mat[0][2]],
                    [mat[1][0], mat[1][1], mat[1][2]],
                    [mat[2][0], mat[2][1], mat[2][2]]])
    q = mat2quat(rotation_mat)
    return np.array([mat[0][3], mat[1][3], mat[2][3], q[0], q[1], q[2], q[3]])

def pose_array_rot_vec_2_pose_array_quat(rot_vec:np.ndarray):
    '''transform pose array of euler to pose array of quaternion

    Param:
        rot_vec:    6d vector, with t(3d) + r(3d)
    ----------
    Return:
        pose:       7d vector, with t(3d) + q(4d)
    '''
    if rot_vec.shape != (6,): raise ValueError
    q = axangle2quat(rot_vec[3:6], np.linalg.norm(rot_vec[3:6]))
    return np.array([rot_vec[0], rot_vec[1], rot_vec[2], q[0], q[1], q[2], q[3]])

def pose_array_euler_2_pose_array_quat(euler:np.ndarray):
    '''transform pose array of euler to pose array of quaternion

    Param:
        euler:      6d vector, with t(3d) + r(3d)
    ----------
    Return:
        pose:       7d vector, with t(3d) + q(4d)
    '''
    if euler.shape != (6,): raise ValueError
    q = euler2quat(euler[3], euler[4], euler[5])
    return np.array([euler[0], euler[1], euler[2], q[0], q[1], q[2], q[3]])

def pose_array_quat_2_pose_array_rot_vec(pos:np.ndarray):
    '''transform pose array of quaternion to pose array of euler

    Param:
        pose:       7d vector, with t(3d) + q(4d)
    ----------
    Return:
        rot_vec:    6d vector, with t(3d) + r(3d)
    '''
    if pos.shape != (7,): raise ValueError
    theta, rot_vec = quat2axangle(pos[3:7])
    k = theta / np.linalg.norm(rot_vec)
    for i, item in enumerate(rot_vec):
        rot_vec[i] = item * k
    return np.array([pos[0], pos[1], pos[2], rot_vec[0], rot_vec[1], rot_vec[2]])

def pose_array_quat_2_pose_array_euler(pos:np.ndarray):
    '''transform pose array of quaternion to pose array of euler

    Param:
        pose:       7d vector, with t(3d) + q(4d)
    ----------
    Return:
        euler:      6d vector, with t(3d) + r(3d)
    '''
    if pos.shape != (7,): raise ValueError
    ai, aj, ak = quat2euler(pos[3:7])
    return np.array([pos[0], pos[1], pos[2], ai, aj, ak])

def pose_array_rotvec_2_matrix(pose:np.ndarray):
    if pose.shape != (6,): raise ValueError
    mat = np.array(cv2.Rodrigues(pose[3:6])[0]).astype(np.float32)
    return np.array([
        [mat[0][0],     mat[0][1],      mat[0][2],      pose[0]],
        [mat[1][0],     mat[1][1],      mat[1][2],      pose[1]],
        [mat[2][0],     mat[2][1],      mat[2][2],      pose[2]],
        [0,             0,              0,              1      ]
    ])

def calc_base_world_mat(world_camera_mat:np.ndarray, tcp_base_pose_quat:np.ndarray, tcp_camera_mat:np.ndarray):
    '''calculate the base's pose relative to the marker
    
    Params:
    ----------
        world_camera_mat:                   a 4x4 matrix, the extrinsic matrix of the in-hand camera
        tcp_base_pose_quat:                 a 7d vector, with t(3d) + q(4d), tcp pose in base coord
        tcp_camera_mat:                     a 4x4 matrix, calibrated before
        
    Return:
    ----------
        base_world_mat:                     a 4x4 matrix, the base's pose relative to the world
    '''
    if world_camera_mat.shape != (4, 4) or tcp_base_pose_quat.shape != (7,) or tcp_camera_mat.shape != (4, 4): raise ValueError
    return np.linalg.inv(world_camera_mat) @ tcp_camera_mat @ np.linalg.inv(pose_array_quat_2_matrix(tcp_base_pose_quat))

def calc_tcp_world_mat(tcp_base_pose_quat:np.ndarray, base_world_mat:np.ndarray):
    """calculate the tcp's pose relative to the marker
    Params:
    ---------
        tcp_base_mat:       a 7d vector, with t(3d) + q(4d)
        base_world_mat:     the base-world matrix obtained from calc_base_world_mat() \
            at calibratin time
            
    Returns:
    ----------
        tcp_world_mat:      a 1x4x4 matrix, the tcp's pose relative to the world
    """
    if tcp_base_pose_quat.shape != (7,) or base_world_mat.shape != (4, 4): raise ValueError
    return base_world_mat @ pose_array_quat_2_matrix(tcp_base_pose_quat)

def calc_tcp_camera_mat(tcp_world_mat:np.ndarray, camera_extrinsics:np.ndarray):
    """calculate the tcp's pose relative to the camera
    Params:
    ----------
        tcp_world_mat:          a 4x4 matrix, tcp's pose relative to world
        camera_extrinsics:      a 4x4 matrix, camera's extrinsics, in the form of R,T,0,1
        
    Returns:
    ----------
        tcp_camera_mat:         a 4x4 matrix, tcp's pose relative to the camera
    """
    if tcp_world_mat.shape != (4, 4) or camera_extrinsics.shape != (4, 4): raise ValueError
    return camera_extrinsics @ tcp_world_mat