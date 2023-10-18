# RH20T Dataset API and Visualizer Implementation

## Dataset API

### Getting Started

The RH20T Python API module is implemented in `rh20t_api` folder. To install the dependencies, run the following command:

```bash
pip install -r ./requirements_api.txt
```

The information for each robot configuration are in `configs/configs.json` file, which is usually required for using the API.

### Basic Usage

Here presents basic usage of the dataset API, including a data extraction script, a scene data loader for loading preprocessed scene data from the dataset, as well as an online preprocessor for real robotic manipulation inference. You can also refer to our visualizer implementation for better understanding how to use the scene data loader.

#### Data Extraction Script

The RGB-D data is stored in mp4 format, we need to change the data into the original image format. In `rh20t_api/extract.py` we provide APIs and a multiprocessing script to convert the dataset into the original image version. Notice that the following scene data loader is based on the original image version of RH20T.

#### Scene Data Loader

The scene data loader should be initialized with a specific scene data folder and robot configurations.

```python
from rh20t_api.configurations import load_conf
from rh20t_api.scene import RH20TScene

robot_configs = load_conf("configs/configs.json")
scene = RH20TScene(scene_path, robot_configs)
```

Some of the methods/properties in the scene data loader that may be of use are listed in the following:

|Method/Property|Comment|
|---|---|
|`RH20TScene.extrinsics_base_aligned`|The preprocessed extrinsics 4x4 matrices for each camera related to robot arm base|
|`RH20TScene.folder`|The current scene folder (can be modified)|
|`RH20TScene.is_high_freq`|Toggles reading high frequency data, default to False|
|`RH20TScene.intrinsics`|Dict[str, np.ndarray] type of camera serial : calibrated 3x4 intrinsic matrices|
|`RH20TScene.in_hand_serials`|The list of in-hand camera serials|
|`RH20TScene.serials`|The list of all camera serials|
|`RH20TScene.low_freq_timestamps`|The list of sorted low-frequency timestamps for each camera serial|
|`RH20TScene.high_freq_timestamps`|The list of sorted high-frequency timestamps (different cameras share the same high-frequency timestamp list)|
|`RH20TScene.start_timestamp`|The starting timestamp for the current scene|
|`RH20TScene.end_timestamp`|The ending timestamp for the current scene|
|`RH20TScene.get_audio_path()`|The audio path|
|`RH20TScene.get_image_path_pairs(timestamp:int, image_types:List[str]=["color", "depth"])`|Query interpolated `Dict[str, List[str]]` type of color-depth image pairs paths for each camera given a timestamp|
|`RH20TScene.get_image_path_pairs_period(time_interval:int, start_timestamp:int=None, end_timestamp:int=None)`|Query a list of interpolated `Dict[str, List[str]]` type of color-depth image pairs paths for each camera given a period of time in milliseconds (starting and ending timestamps will be set to the scene's if not specified)|
|`RH20TScene.get_ft_aligned(timestamp:int, serial:str="base", zeroed:bool=True)`|Query interpolated preprocessed force-torque concatenated 6d vector given a timestamp and a camera serial (or "base" which reads data from all serials)|
|`RH20TScene.get_tcp_aligned(timestamp:int, serial:str="base")`|Query interpolated preprocessed tcp 7d quaternion pose vector given a timestamp and a camera serial (or "base" which reads data from all serials)|
|`RH20TScene.get_joint_angles_aligned(timestamp:int, serial:str="base")`|Query interpolated joint angles sequence given a timestamp and a camera serial (or "base" which reads data from all serials)|
|`RH20TScene.get_gripper_command(timestamp:int)`|Query interpolated gripper command given a timestamp|
|`RH20TScene.get_gripper_info(timestamp:int)`|Query interpolated gripper info given a timestamp|

#### Online Preprocessor

The force and torque concatenated vectors, as well as the tcp values are collected in robot arm base coordinate. The implemented online preprocessor can be used to project these values to a certain camera coordinate online for inferencing purpose. It should be initialized with calibration result path, and the specific camera to project to.

```python
from rh20t_api.configurations import load_conf
from rh20t_api.scene import RH20TScene
from rh20t_api.online import RH20TOnline

serial = "[The serial number of the camera to project to]"

robot_configs = load_conf("configs/configs.json")
scene = RH20TScene(scene_path, robot_configs)

# before using the processor, it is recommended to collect the first several frames of data 
# when the robot arm is still, and update the sensor offsets considering the temperature drift
sampled_raw_fts = # shaped (n, 6), raw force-torque vectors in the first several frames
sampled_raw_tcps = # shaped (n, 6) or (n, 7), raw tcp values in the first several frames
scene.configuration.update_offset(sampled_raw_fts, sampled_raw_tcps)

# initialize the preprocessor
processor = RH20TOnline(scene.calib_folder, scene.configuration, serial)

# ...

# online preprocessing ft_raw and tcp_raw, the processed values are 
# aligned, processed with sensor offsets (for force and torque) and 
# projected to the specified camera
processed_ft = processor.project_raw_from_external_sensor(ft_raw, tcp_raw)
processed_tcp, processed_ft_tcp = processor.project_raw_from_robot_sensor(tcp_raw)
```

## Dataset Visualizer

### Getting Started

The visualizer can be configured in `configs/default.yaml`. The dependencies are installed via:

```bash
# Recommend running on Ubuntu
pip install -r ./requirements.txt
```

The minimal files requirement for visualizing a scene will be a scene folder placed together with a calibration folder like:

```text
- calib
    - [some timestamp]
    - ...
- [scene folder]
```

Before visualizing a scene, we should first preprocess the images to point clouds and cache them to save time, for example:

```bash
python visualize.py --scene_folder [SCENE_FOLDER] --cache_folder [CACHE_FOLDER] --preprocess
```

**NOTE: The RGB and depth images were originally 1280x720 and are now resized to 640x360. To get the visualization run normally, we either need to resize the images back to 1280x720 or modify the camera intrinsics. In this implementation, we explicitly scale the intrinsics by 0.5 at [here](https://github.com/rh20t/rh20t_api/blob/main/utils/point_cloud.py#L93-L94). If you are projecting the depth to point cloud in your own project, don't forget this step!**

You can modify configurations including sampling time interval, screenshot saving path, and choices to enable visualizing etc in `configs/default.yaml`. If you would only like to view the first several frames of point clouds, you can run the above command for a while and then stop it with `Ctrl + C`. It is recommended to cache at least the first frame point cloud since it is used for adjusting the initial viewing direction.

Running the following command will visualize the dynamic scene, with an Open3D visualizer viewport showing the 3D dynamic scene, and an OpenCV viewport showing the real-world video captured by your chosen camera (determined by `chosen_cam_idx` in `configs/default.yaml`). Note that the scene folder and cache folder should match the previous ones.

```bash
python visualize.py --scene_folder [SCENE_FOLDER] --cache_folder [CACHE_FOLDER]
```

During the visualization, you can:

1. drag and view the scene with your mouse;
2. press `Alt` to pause, then press `←` and `→` to view previous and next frames respectively, drag and view them, press `C` to save a screenshot of the current frame, and press `Alt` again for continue playing;
3. press `Esc` to stop.
