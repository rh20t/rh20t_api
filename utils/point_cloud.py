import logging
import os
from typing import Any, Dict

import cv2
import numpy as np
import open3d as o3d
from tqdm import tqdm


class PointCloud:
    # TODO: L515
    def __init__(
        self, 
        logger:logging.Logger,
        downsample_voxel_size_m:float=0.0001, 
        filter_num_neighbor:int=10, 
        filter_radius_m:float=0.01, 
        filter_std_ratio:float=2.0, 
        normal_radius:float=0.01, 
        normal_num_neighbor:int=30, 
        min_depth_m:float=0.3, 
        max_depth_m:float=0.8, 
        width:int=640, 
        height:int=360,
        pcd_file_format:str=".ply",
        debug:bool=True
    ):
        self.downsample_voxel_size_m = downsample_voxel_size_m
        self.filter_num_neighbor = filter_num_neighbor
        self.filter_std_ratio = filter_std_ratio
        self.normal_radius = normal_radius
        self.normal_num_neighbor = normal_num_neighbor
        self.min_depth_m = min_depth_m
        self.max_depth_m = max_depth_m
        self.width = width
        self.height = height
        self.filter_radius_m = filter_radius_m
        self.debug = debug
        self.logger = logger
        self.pcd_file_format = pcd_file_format

        self.img_names = {}

    def merge_pointclouds(self, pcd1:o3d.geometry.PointCloud, pcd2:o3d.geometry.PointCloud):
        merged_points = np.vstack((np.asarray(pcd1.points), np.asarray(pcd2.points)))
        merged_colors = np.vstack((np.asarray(pcd1.colors), np.asarray(pcd2.colors)))

        merged_pcd = o3d.geometry.PointCloud()
        merged_pcd.points = o3d.utility.Vector3dVector(merged_points)
        merged_pcd.colors = o3d.utility.Vector3dVector(merged_colors)

        # merged_pcd = merged_pcd.voxel_down_sample(self.downsample_voxel_size_m)

        _, ind = merged_pcd.remove_statistical_outlier(nb_neighbors=self.filter_num_neighbor, 
            std_ratio=self.filter_std_ratio)
        merged_pcd = merged_pcd.select_by_index(ind)

        merged_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=self.normal_radius, max_nn=self.normal_num_neighbor))

        return merged_pcd
    
    def rgbd_to_pointcloud(
        self, 
        is_l515:bool,
        color_image_path:str, 
        depth_image_path:str, 
        width:int, 
        height:int, 
        intrinsic:np.ndarray, 
        extrinsic:np.ndarray=np.eye(4), 
        downsample_factor:float=1
    ):
        color = cv2.cvtColor(cv2.imread(color_image_path), cv2.COLOR_BGR2RGB)
        depth = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED).astype(np.float32)

        # downsample image
        color = cv2.resize(color, (int(width / downsample_factor), int(height / downsample_factor))).astype(np.int8)
        depth = cv2.resize(depth, (int(width / downsample_factor), int(height / downsample_factor)))

        depth /= 4000.0 if is_l515 else 1000.0  # from millimeters to meters
        depth[depth < self.min_depth_m] = 0
        depth[depth > self.max_depth_m] = 0
        

        rgbd_image = o3d.geometry.RGBDImage()
        rgbd_image = rgbd_image.create_from_color_and_depth(o3d.geometry.Image(color),
            o3d.geometry.Image(depth), depth_scale=1.0, convert_rgb_to_intensity=False)

        intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic()
        intrinsic_o3d.set_intrinsics(int(width / downsample_factor), int(height / downsample_factor), 
            0.5 * intrinsic[0, 0] / downsample_factor, 0.5 * intrinsic[1, 1] / downsample_factor, 
            0.5 * intrinsic[0, 2] / downsample_factor, 0.5 * intrinsic[1, 2] / downsample_factor)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic_o3d, extrinsic=extrinsic)
        return pcd

    def filter_pointclouds(self, pcds, num_neighbors:int, std_ratio:float, radius):
        for i in range(len(pcds)):
            _, ind = pcds[i].remove_statistical_outlier(nb_neighbors=num_neighbors, std_ratio=std_ratio)
            pcds[i] = pcds[i].select_by_index(ind)
            if radius > 0:
                _, ind = pcds[i].remove_radius_outlier(nb_points=num_neighbors, radius=radius)
                pcds[i] = pcds[i].select_by_index(ind)

        return pcds
    
    def point_cloud_path(self, write_folder:str, timestamp:int):
        return os.path.join(write_folder, str(timestamp) + self.pcd_file_format)

    def point_cloud_single_frame(
        self, 
        image_pairs:dict, 
        timestamp:int, 
        in_hand_serials:list, 
        intrinsics:dict, 
        extrinsics:dict, 
        write_folder:str=""
    ):
        """generate a point cloud according to multiview RGBD image pairs for single frame

        Args:
            image_pairs (dict):             color and depth image path pairs
            timestamp (int):                _description_
            in_hand_serials (list):         the serial numbers of the in-hand cameras
            intrinsics (np.array):          dict of 3x4 numpy matrices of intrinsics
            extrinsics (np.array):          dict of 4x4 numpy matrices of extrinsics
            write_folder (str, optional):   the path to write the generated point cloud, "" for not to write. Defaults to "".

        Returns:
            o3d.geometry.PointCloud:        the generated point cloud
        """
        # generate point clouds
        pcds = []
        for serial in image_pairs:
            if serial in in_hand_serials or len(image_pairs[serial]) == 0:
                if self.debug: self.logger.debug(f"[PointCloud] Met in-hand camera {serial}")
                continue
            pcds.append(
                self.rgbd_to_pointcloud(
                    serial[0].isalpha(),
                    image_pairs[serial][0], 
                    image_pairs[serial][1],
                    self.width,
                    self.height,
                    intrinsics[serial],
                    extrinsics[serial]
                )
            )

        # merge the clouds and visualize
        pcds = [pcd.voxel_down_sample(self.downsample_voxel_size_m) for pcd in pcds]
        pcds = self.filter_pointclouds(pcds, self.filter_num_neighbor, self.filter_std_ratio, self.filter_radius_m)
        full_pcd = pcds[0]
        for i in range(1, len(pcds)):
            if i not in [2, 3]: continue
            full_pcd = self.merge_pointclouds(pcds[i], full_pcd)
            
        if self.debug: self.logger.info(f"pcd size: {np.asarray(full_pcd.points).shape}")

        if write_folder != "":
            os.makedirs(write_folder, exist_ok=True)
            o3d.io.write_point_cloud(self.point_cloud_path(write_folder, timestamp), full_pcd, write_ascii=False)
        else:
            if self.debug: self.logger.info("write folder path is empty, not writing!") 

        _, ind = full_pcd.remove_statistical_outlier(nb_neighbors=self.filter_num_neighbor, 
            std_ratio=self.filter_std_ratio)
        full_pcd.select_by_index(ind)

        return full_pcd
    
    def point_cloud_multi_frames(self, image_pairs:list, in_hand_serials:list, intrinsics:np.ndarray, extrinsics:np.ndarray, write_folder:str=""):
        """generate point clouds according to multiview RGBD image pairs for multiple frames

        Args:
            image_pairs (list):             color and depth image path pairs
            in_hand_serials (list):         the serial numbers of the in-hand cameras
            intrinsics (np.array):          dict of 3x4 numpy matrices of intrinsics
            extrinsics (np.array):          dict of 4x4 numpy matrices of extrinsics
            write_folder (str, optional):   the path to write the generated point cloud, "" for not to write. Defaults to "".

        Returns:
            o3d.geometry.PointCloud:        the generated point clouds if write_folder is "", else None
        """
        pcds = [] if write_folder == "" else None
        for img_idx, img_pair in enumerate(tqdm(image_pairs)):
            _t = img_pair["timestamp"]
            del img_pair["timestamp"]
            if pcds:
                pcds.append(
                    self.point_cloud_single_frame(
                        image_pairs=img_pair,
                        timestamp=_t,
                        in_hand_serials=in_hand_serials,
                        intrinsics=intrinsics,
                        extrinsics=extrinsics,
                        write_folder=write_folder
                    )   
                )
            else:
                self.point_cloud_single_frame(
                    image_pairs=img_pair,
                    timestamp=_t,
                    in_hand_serials=in_hand_serials,
                    intrinsics=intrinsics,
                    extrinsics=extrinsics,
                    write_folder=write_folder
                )                  
        return pcds
    

def create_point_cloud_manager(logger:logging.Logger, vis_cfg:Dict[str, Any]):
    """
        Create a point cloud manager given a configuration.
        
        Params:
        ----------
            logger:     the point cloud logger
            vis_cfg:    configuration
        
        Returns:
        ----------
            pointcloud: created point cloud manager
    """
    return PointCloud(
        logger=logger, 
        downsample_voxel_size_m=vis_cfg["downsample_voxel_size_m"],
        filter_num_neighbor=vis_cfg["filter_num_neighbor"], 
        filter_radius_m=vis_cfg["filter_radius_m"],
        filter_std_ratio=vis_cfg["filter_std_ratio"], 
        normal_radius=vis_cfg["normal_radius"],
        normal_num_neighbor=vis_cfg["normal_num_neighbor"], 
        min_depth_m=vis_cfg["min_depth_m"],
        max_depth_m=vis_cfg["max_depth_m"], 
        width=vis_cfg["resolution"][0], 
        height=vis_cfg["resolution"][1],
        debug=vis_cfg["debug"]
    )
