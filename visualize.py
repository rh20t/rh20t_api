import argparse
from logging import Logger
import os
import queue
from alive_progress import alive_bar
import cv2
from matplotlib import pyplot as plt
import yaml
import time
import kinpy as kp
import open3d as o3d
import numpy as np
from rh20t_api.configurations import load_conf, tcp_as_q, Configuration
from rh20t_api.convert import timestamp_to_datetime_str
from rh20t_api.online import aligned_tcp_glob_mat, zeroed_force_torque_base
from rh20t_api.scene import RH20TScene
from utils.keyboard_listener import KeyboardListener
from utils.logger import logger_begin
from utils.point_cloud import create_point_cloud_manager
from utils.stopwatch import Stopwatch
from utils.robot import RobotModel
import librosa
import librosa.display
from typing import Dict, Any
import traceback

def get_trans_mat(start_coord:np.ndarray, end_coord:np.ndarray):
    """
        Obtain the transformation matrix and scale given starting and ending coords.
        
        This is used for transforming line segments or arrow meshes to the desired 
            starting and ending coords.
        
        Params:
        ----------
            start_coord:    starting coordinate shaped (3,)
            end_coord:      ending coordinate shaped (3,)
        
        Returns:
        ----------
            trans_mat:      4x4 transformation matrix
            scale:          the scale of the desired line segment/arrow
    """
    _scale:float = np.linalg.norm(end_coord - start_coord)
    [_x, _y, _z] = ((end_coord - start_coord) / _scale).tolist()
    return np.array([
            [1. / (1 + (_x / _z) * (_x / _z)),  0.,                                 _x, start_coord[0]], 
            [0.,                                1. / (1 + (_y / _z) * (_y / _z)),   _y, start_coord[1]], 
            [-(_x * _z) / (_x * _x + _z * _z),  -(_y * _z) / (_y * _y + _z * _z),   _z, start_coord[2]], 
            [0.,                                0.,                                 0., 1.]]), _scale

def force_torque_split(tcp_aligned:np.ndarray, force_torque_aligned_zeroed:np.ndarray, ratio_f:float=0.001, ratio_t:float=0.001):
    _coord_start, _force, _torque = tcp_aligned[0:3], force_torque_aligned_zeroed[0:3], force_torque_aligned_zeroed[3:6]
    return _coord_start, _coord_start + _force * ratio_f, _coord_start + _torque * ratio_t, np.linalg.norm(_force), np.linalg.norm(_torque)

def create_robot_model_manager(configuration:Configuration):
    return RobotModel(configuration.robot_joint_sequence, configuration.robot_urdf, configuration.robot_mesh)

def create_traj_mesh(prev_coord:np.ndarray, now_coord:np.ndarray):
    """
        Visualize a segment of trajectory.
        
        Params:
        ----------
            prev_coord:     the previous coordinate
            now_coord:      the current coordinate
            
        Returns:
        ----------
            traj_mesh:      the created trajectory mesh
    """
    _trans, _scale = get_trans_mat(prev_coord, now_coord)
    traj_mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=0.25, height=1.0, resolution=20, 
        split=4, create_uv_map=False).scale(_scale, [0,0,0]).transform(_trans).paint_uniform_color([0., 1., 0.])
    traj_mesh.compute_vertex_normals()
    return traj_mesh

def renderer_update(visualizer):
    visualizer.poll_events()
    visualizer.update_renderer()

class AudioManager:
    def __init__(self, scene:RH20TScene, vis_cfg:Dict[str, Any]) -> None:
        """
            Audio data manager; audio data will be visualized in waveform.
            
            Params:
            ----------
                scene:      the current scene instance
        """
        self._scene = scene
        self._vis_cfg = vis_cfg
        
        self._audio_path = scene.get_audio_path()
        self._audio_data, self._sample_rate = librosa.load(self._audio_path)
        self._audio_window_size = self._audio_data.shape[0] // ((scene.end_timestamp + 1 - scene.start_timestamp) // vis_cfg['time_interval'])
        
    def save(self, t:float, save_folder:str):
        """
            Save the waveform audio data at time t
            
            Params:
            ----------
                t:              the time of audio data to save
                save_folder:    the folder to save the waveform
        """
        _s = (t - self._scene.start_timestamp) // self._vis_cfg['time_interval']
        plt.clf()
        librosa.display.waveshow(self._audio_data[int((_s - 0.5) * self._audio_window_size):
            int((_s + 0.5) * self._audio_window_size)], sr=self._sample_rate)
        plt.ylim(-0.015, 0.015)
        plt.xlabel("")
        plt.savefig(os.path.join(save_folder, "wave.png"))
        plt.xticks([])
        plt.yticks([])
        plt.savefig(os.path.join(save_folder, "wave_no_axis.png"))
        

class SingleViewCameraManger:
    def __init__(self, vis_cfg:Dict[str, Any], scene:RH20TScene, chosen_cam_id:int, logger:Logger) -> None:
        """
            Managing single view rgb image, choose one camera index and visualize its RGB image.
            
            Params:
            ----------
                vis_cfg:        the visualization configuration
                scene:          current scene instance
                chosen_cam_id:  the chosen camera index, can be 0, 1, ...
                logger:         the logger for the manager
        """
        self._vis_cfg = vis_cfg
        self._scene = scene
        self._chosen_cam_id = chosen_cam_id
        self._logger = logger
        
        self._create_chosen_camera()
    
    def _create_chosen_camera(self):
        """
            Given the chosen index of camera, create the corresponding camera for viewing the scene.
        """
        self._extrinsics = self._scene.extrinsics_base_aligned
        self._intrinsics = self._scene.intrinsics
        
        cams_serial = list(self._scene.extrinsics_base_aligned.keys())
        self._chosen_serial = cams_serial[self._chosen_cam_id]
        self._logger.info(f"All camera serials: {cams_serial}\nChosen camera: {self._chosen_serial}")

        self._camera = o3d.camera.PinholeCameraParameters()
        self._camera.extrinsic = self._extrinsics[cams_serial[self._chosen_cam_id]]
        chosen_intrinsic = self._intrinsics[cams_serial[self._chosen_cam_id]]
        self._camera.intrinsic = o3d.camera.PinholeCameraIntrinsic(
            #TODO: check for L515 case, which may not be in this resolution
            width=self._vis_cfg['resolution'][0], 
            height=self._vis_cfg['resolution'][1], 
            fx=chosen_intrinsic[0][0], 
            fy=chosen_intrinsic[1][1], 
            cx=chosen_intrinsic[0][2], 
            cy=chosen_intrinsic[1][2]
        )
    
    def init_window(self, width=320, height=180):
        """
            Initialize visualization window
            
            Params:
            ----------
                width:      the viewport width
                height:     the viewport height
        """
        self._window_name = f"{self._chosen_serial} Capture"
        cv2.namedWindow(self._window_name, 0)
        cv2.resizeWindow(self._window_name, width, height)
    
    def create_camera_models(self):
        """
            Create camera meshes for visualization.
            
            Returns:
            ----------
                models:     a list of the create camera meshes
        """
        models = []
        for k in self._extrinsics:
            if k in self._scene.in_hand_serials: continue
            cam_mesh = o3d.geometry.LineSet.create_camera_visualization(
                view_width_px=1280, 
                view_height_px=720, 
                intrinsic=self._intrinsics[k][:3, :3], 
                extrinsic=self._extrinsics[k], 
                scale=0.04
            ).paint_uniform_color([1., 0., 0.])
            models.append(cam_mesh)
        return models
    
    def update_img(self, img_path:str):
        """
            Update the displayed image.
            
            Params:
            ----------
                img_path:   the image path to read in
        """
        self._img = cv2.imread(img_path)
        cv2.imshow(self._window_name, self._img)
        cv2.waitKey(1)
    
    def save(self, save_folder:str): cv2.imwrite(os.path.join(save_folder, "cam.png"), self._img)
    
    @property
    def camera(self): return self._camera
    
    @property
    def serial(self): return self._chosen_serial
        
        
class FTArrowManager:
    def __init__(self) -> None:
        """
            Managing the visualization of arrow meshes representing force and torque
        """
        cylinder_radius=2.0
        cone_radius=1.5
        cylinder_height=5.0 
        cone_height=4.0
        resolution=20
        cylinder_split=4
        cone_split=1
        
        self._mesh_arrow_f = o3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=cylinder_radius, cone_radius=cone_radius, cylinder_height=cylinder_height, 
            cone_height=cone_height, resolution=resolution, cylinder_split=cylinder_split, 
            cone_split=cone_split).paint_uniform_color([1., 0., 0.])
        self._mesh_arrow_t = o3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=cylinder_radius, cone_radius=cone_radius, cylinder_height=cylinder_height, 
            cone_height=cone_height, resolution=resolution, cylinder_split=cylinder_split, 
            cone_split=cone_split).paint_uniform_color([0., 0., 1.])
        
        self._prev_mesh_arrow_f_trans, self._prev_mesh_arrow_t_trans = None, None
        self._prev_f_scale, self._prev_t_scale = 1., 1.
    
    def update(self, ft_start:np.ndarray, f_end:np.ndarray, t_end:np.ndarray):
        """
            Update the arrows according to the starting and ending points coordinates.
            
            Params:
            ----------
                ft_start:   the starting point, i.e., the gripper location
                f_end:      the ending point for the force vector
                t_end:      the ending point for the torque vector
        """
        if self._prev_mesh_arrow_f_trans is not None: self._mesh_arrow_f.transform(np.linalg.inv(self._prev_mesh_arrow_f_trans))
        self._prev_mesh_arrow_f_trans, _scale = get_trans_mat(ft_start, f_end)
        self._mesh_arrow_f.scale(_scale / self._prev_f_scale, [0,0,0])
        self._prev_f_scale = _scale
        self._mesh_arrow_f.transform(self._prev_mesh_arrow_f_trans)
        self._mesh_arrow_f.compute_vertex_normals()
        if self._prev_mesh_arrow_t_trans is not None: self._mesh_arrow_t.transform(np.linalg.inv(self._prev_mesh_arrow_t_trans))
        self._prev_mesh_arrow_t_trans, _scale = get_trans_mat(ft_start, t_end)
        self._mesh_arrow_t.scale(_scale / self._prev_t_scale, [0,0,0])
        self._prev_t_scale = _scale
        self._mesh_arrow_t.transform(self._prev_mesh_arrow_t_trans)
        self._mesh_arrow_t.compute_vertex_normals()
    
    @property
    def geometries(self): return [self._mesh_arrow_f, self._mesh_arrow_t]
        

def save_results(
    t:float, scene_path:str, visualizer, screenshot_path:str, audio_manager:AudioManager, 
    single_view_cam:SingleViewCameraManger, logger:Logger
):
    """
        Save the results of the current frame.
        
        Params:
        ----------
            t:                  current timr
            scene_path:         the path to the scene folder
            visualizer:         current visualizer instance
            screenshot_path:    the path to save screenshot
            audio_manager:      audio data writer
            single_view_cam:    cam data writer
            logger:             the logger for the saving process
    """
    logger.info("pressed S, saving")
    _save_folder = os.path.join(screenshot_path, str(int(time.time() * 100)))
    os.makedirs(_save_folder, exist_ok=True)
    audio_manager.save(t, _save_folder)
    _save_path_model = os.path.join(_save_folder, scene_path.split('/')[-1] + ".png")
    visualizer.capture_screen_image(_save_path_model, do_render=True)
    single_view_cam.save(_save_folder)
    
    logger.info(f"saved screenshot in {_save_folder}")

def visualize(scene_path:str, pcd_folder:str, vis_cfg:dict, logger:Logger):
    max_traj_size = vis_cfg['max_traj_size']
    screenshot_path = vis_cfg['screenshot_path']
    _width, _height = vis_cfg['viewport_width'], vis_cfg['viewport_height']
    chosen_cam_idx = vis_cfg['chosen_cam_idx']
    enable_ft = vis_cfg["enable_ft"]
    enable_pcd = vis_cfg["enable_pcd"]
    enable_model = vis_cfg["enable_model"]
    enable_traj = vis_cfg["enable_traj"]

    robot_configs = load_conf(vis_cfg["robot_configs"])
    dataloader = RH20TScene(scene_path, robot_configs)
    logger.info(f"Calibration quality of this scene: {dataloader.metadata['calib_quality']}")

    pointcloud = create_point_cloud_manager(logger, vis_cfg)

    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window(width=_width, height=_height)

    if enable_model: robot_model = create_robot_model_manager(dataloader.configuration)
    start_timestamp, end_timestamp = dataloader.start_timestamp, dataloader.end_timestamp

    pcd, ft_start = None, None
    _n_frames, _vis_time = 0, 0.
    _t = start_timestamp
    first_time = True
    traj_queue = queue.Queue(maxsize=0)

    logger.info('======Press ESC to exit, or Alt to pause=====')

    stopwatch = Stopwatch()
    keyboard_listener = KeyboardListener()
    keyboard_listener.start()
    
    ft_arrow = FTArrowManager()
    audio = AudioManager(dataloader, vis_cfg)
    single_view_cam = SingleViewCameraManger(vis_cfg, dataloader, chosen_cam_idx, logger)
    camera_models = single_view_cam.create_camera_models()
    for cam_mesh in camera_models: visualizer.add_geometry(cam_mesh, reset_bounding_box=False)
    
    ctr = visualizer.get_view_control()
    ctr.convert_from_pinhole_camera_parameters(single_view_cam.camera, allow_arbitrary=True)

    single_view_cam.init_window()

    try:
        with alive_bar(end_timestamp + 1 - start_timestamp, manual=True, title='3D scene') as bar:
            while _t < end_timestamp + 1:
                img_path = dataloader.get_image_path_pairs(_t, image_types=["color"])[single_view_cam.serial][0]
                single_view_cam.update_img(img_path)
                if keyboard_listener.esc:
                    logger.info('Pressed esc, exitting...')
                    break
                if keyboard_listener.pause:
                    # saving is only enabled when paused
                    if keyboard_listener.save:
                        save_results(_t, scene_path, visualizer, screenshot_path, audio, single_view_cam, logger)
                        keyboard_listener.save = False
                    # checking previous and next frames is only enabled when paused
                    if keyboard_listener.left == 0: 
                        renderer_update(visualizer)
                        continue
                    _t -= (keyboard_listener.left + 1) * vis_cfg['time_interval']
                    keyboard_listener.left = 0
                    if start_timestamp > _t: _t = start_timestamp 
                    if end_timestamp < _t: _t = end_timestamp
                
                stopwatch.reset()

                if enable_model:
                    # visualize robot arm model
                    robot_model.update(dataloader.get_joint_angles_aligned(_t), first_time)
                    for m in robot_model.geometries_to_add: visualizer.add_geometry(m, reset_bounding_box=False)
                    for m in robot_model.geometries_to_update: visualizer.update_geometry(m)                    
                    renderer_update(visualizer)

                # tcp
                tcp_aligned = dataloader.get_tcp_aligned(_t)
                if ft_start is not None: prev_coord = ft_start.copy()
                force_torque_preprocessed = dataloader.get_ft_aligned(_t)
                ft_start, f_end, t_end, _, _ = force_torque_split(tcp_aligned, force_torque_preprocessed)
                if enable_ft:
                    # visualize force and torque with arrows
                    ft_arrow.update(ft_start, f_end, t_end)
                    for mesh_arrow in ft_arrow.geometries:
                        if first_time: visualizer.add_geometry(mesh_arrow, reset_bounding_box=False)
                        else: visualizer.update_geometry(mesh_arrow)
                    renderer_update(visualizer)
                # trajectory
                if enable_traj and not first_time:
                    traj_mesh = create_traj_mesh(prev_coord, ft_start)
                    visualizer.add_geometry(traj_mesh, reset_bounding_box=False)
                    traj_queue.put(traj_mesh)
                    if max_traj_size > 0:
                        while not keyboard_listener.pause and traj_queue.qsize() > max_traj_size:
                            _line = traj_queue.get()
                            visualizer.remove_geometry(_line, reset_bounding_box=False)
                    renderer_update(visualizer)

                # point cloud
                if pcd: visualizer.remove_geometry(pcd, reset_bounding_box=False)
                if first_time or enable_pcd:
                    pcd = o3d.io.read_point_cloud(pointcloud.point_cloud_path(pcd_folder, _t))
                    bounding_box = o3d.geometry.AxisAlignedBoundingBox(min_bound=[-0.15, -0.5, -0.1], max_bound=[0.9, 0.5, 0.9])
                    pcd = pcd.crop(bounding_box)
                    visualizer.add_geometry(pcd, reset_bounding_box=first_time)
                    renderer_update(visualizer)
    
                if first_time: first_time = False
                else:
                    _vis_time += stopwatch.split
                    _n_frames += 1
                _t += vis_cfg['time_interval']
                bar((_t - start_timestamp) / (end_timestamp + 1 - start_timestamp))

        if not keyboard_listener.esc: from IPython.terminal import embed; ipshell=embed.InteractiveShellEmbed(config=embed.load_default_config())(local_ns=locals())        

    except Exception as e:
        logger.error(e)
        traceback.print_exc()
    finally:
        keyboard_listener.terminate()
    
    logger.info(f"Average time: {_vis_time / _n_frames:.3f} s")
    logger.info(f"Average FPS: {_n_frames / _vis_time:.3f}")

def pcd_preprocess(scene_path:str, pcd_folder:str, vis_cfg:dict, logger:Logger):
    robot_configs = load_conf(vis_cfg["robot_configs"])
    dataloader = RH20TScene(scene_path, robot_configs)
    pointcloud = create_point_cloud_manager(logger, vis_cfg)
    stopwatch = Stopwatch()
    pointcloud.point_cloud_multi_frames(
        image_pairs=dataloader.get_image_path_pairs_period(vis_cfg["time_interval"]),
        in_hand_serials=dataloader.in_hand_serials,
        intrinsics=dataloader.intrinsics,
        extrinsics=dataloader.extrinsics_base_aligned,
        write_folder=pcd_folder
    )
    logger.info(f"preprocessing finished in {stopwatch.split} seconds.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene_folder', help='Path to the data directory to visualize', type=str, required=True)
    parser.add_argument('--cache_folder', default="", help='Path to cache the generated point cloud', type=str, required=True)
    parser.add_argument('--preprocess', action='store_true', help='Preprocess and cache the point clouds')
    ARGS = parser.parse_args()
    if ARGS.scene_folder[-1] == "/": ARGS.scene_folder = ARGS.scene_folder[:-1]
    if ARGS.cache_folder[-1] == "/": ARGS.cache_folder = ARGS.cache_folder[:-1]

    vis_logger = logger_begin(name="Vis Logger", color=True, level="DEBUG")

    try:
        with open(os.path.join('configs', 'default.yaml'), 'r') as settings_file: vis_cfg_dict = yaml.load(settings_file, Loader = yaml.FullLoader)
    except: 
        vis_logger.error("No configuration file `./configs/default.yaml` existing!")
        exit(1)
    
    if ARGS.preprocess:
        pcd_preprocess(ARGS.scene_folder, ARGS.cache_folder, vis_cfg_dict, vis_logger)
    else:
        visualize(ARGS.scene_folder, ARGS.cache_folder, vis_cfg_dict, vis_logger)

