import sys
sys.path.append(".")
from rh20t_api.configurations import load_conf
from rh20t_api.scene import RH20TScene
import numpy as np

def test_joint_angle_getter():
    robot_configs = load_conf("configs/configs.json")
    for scene_path in [
        "/aidata/RH20T_resized/RH20T_cfg4/task_0001_user_0010_scene_0001_cfg_0004",
        "/aidata/RH20T_resized/RH20T_cfg1/task_0001_user_0001_scene_0001_cfg_0001"
    ]:
        scene = RH20TScene(scene_path, robot_configs)
        start_t = scene.start_timestamp_low_freq
        end_t = scene.end_timestamp_low_freq
        rnd_t = np.random.randint(start_t + 1, end_t)
        for test_t in [start_t, end_t, rnd_t]:
            # `RH20TScene.get_joints_angles` has been tested in visualization
            # assert np.sum(np.abs(scene.get_joints_angles(test_t) - scene.get_joint_angles_aligned(test_t, serial="base"))) < 1e-9
            print(scene.get_joint_angles_aligned(test_t, serial="base"))

if __name__ == "__main__":
    test_joint_angle_getter()
