import os
from typing import List

import kinpy as kp
import numpy as np
import open3d as o3d


class RobotModel:
    def __init__(self, robot_joint_sequence:List[str], robot_urdf:str, robot_mesh:str) -> None:
        """Robot model manager.

        Params:
        ----------
            robot_joint_sequence:   the robot joint name sequence
            robot_urdf:             the urdf file path for the robot model
            robot_mesh:             the mesh folder for the robot model
        """
        self._robot_joint_sequence = robot_joint_sequence
        self._robot_urdf = robot_urdf
        self._robot_mesh = robot_mesh

        self._model_chain = kp.build_chain_from_urdf(open(robot_urdf).read().encode('utf-8'))
        self._visuals_map = self._model_chain.visuals_map()
        
        self._model_meshes = {}
        self._prev_model_transform = {}
        
        self._init_buffer()
        
    def _init_buffer(self):
        self._geometries_to_add = []
        self._geometries_to_update = []
    
    def update(self, rotates:np.ndarray, first_time:bool):
        """
            Update robot model transformations.
            
            Params:
            ----------
                rotates:        rotations for each joint
                first_time:     if it is the first time when geometries 
                                should be added to the visualizer
            
            Returns:
            ----------
                None
        """
        self._init_buffer()
        transformations = {joint: rotates[i] for i, joint in enumerate(self._robot_joint_sequence)}
        cur_transforms = self._model_chain.forward_kinematics(transformations)
        for link, transform in cur_transforms.items():
            if first_time: self._model_meshes[link], self._prev_model_transform[link] = {}, {}
            for v in self._visuals_map[link]:
                if v.geom_param is None: continue
                tf = np.dot(transform.matrix(), v.offset.matrix())
                if first_time: 
                    self._model_meshes[link][v.geom_param] = o3d.io.read_triangle_mesh(os.path.join(self._robot_mesh, v.geom_param))
                    self._geometries_to_add.append(self._model_meshes[link][v.geom_param])
                else:
                    self._model_meshes[link][v.geom_param].transform(np.linalg.inv(self._prev_model_transform[link][v.geom_param]))
                self._model_meshes[link][v.geom_param].transform(tf)
                self._prev_model_transform[link][v.geom_param] = tf
                self._model_meshes[link][v.geom_param].compute_vertex_normals()
                self._geometries_to_update.append(self._model_meshes[link][v.geom_param])
    
    @property
    def geometries_to_add(self): return self._geometries_to_add
    
    @property
    def geometries_to_update(self): return self._geometries_to_update