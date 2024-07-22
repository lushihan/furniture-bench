import torch
import numpy as np

# from gym import spaces
from gymnasium import spaces
import gymnasium as gym

from furniture_bench.envs.furniture_bench_env import FurnitureBenchEnv
from furniture_bench.config import config
from furniture_bench.perception.image_utils import resize, resize_crop
from furniture_bench.robot.panda import PandaError
from furniture_bench.utils.frequency import set_frequency
from furniture_bench.config import config
from furniture_bench.data.collect_enum import CollectEnum


class FurnitureBenchImageRobomimic(FurnitureBenchEnv):
    """Furniture environment with image observation and force array."""

    def __init__(self, **kwargs):
        super().__init__(
            furniture=kwargs["furniture"],
            # randomness="low",
            randomness="high_collect",
            abs_action=kwargs["abs_action"],
            act_rot_repr=kwargs["act_rot_repr"],
        )

        self.img_shape = (3, *config["furniture"]["env_img_size"])
        self.img_shape = (*config["furniture"]["env_img_size"], 3)
        self.num_envs = 1

    @property
    def observation_space(self):
        low, high = -np.inf, np.inf
        dof = self.robot.dof
        img_size = config["furniture"]["env_img_size"]
        return gym.spaces.Dict(
            {
                "ee_pos": gym.spaces.Box(low=low, high=high, shape=(3,)),  # (x, y, z)
                "ee_quat": gym.spaces.Box(
                    low=low, high=high, shape=(4,)
                ),  #  (x, y, z, w)
                "ee_pos_vel": gym.spaces.Box(low=low, high=high, shape=(3,)),
                "ee_ori_vel": gym.spaces.Box(low=low, high=high, shape=(3,)),
                "joint_positions": gym.spaces.Box(low=low, high=high, shape=(dof,)),
                "joint_velocities": gym.spaces.Box(low=low, high=high, shape=(dof,)),
                "joint_torques": gym.spaces.Box(low=low, high=high, shape=(dof,)),
                "gripper_width": gym.spaces.Box(low=low, high=high, shape=(1,)),
                "color_image1": gym.spaces.Box(low=0, high=255, shape=(*img_size, 3)),
                "color_image2": gym.spaces.Box(low=0, high=255, shape=(*img_size, 3)),
                # "active_acous": gym.spaces.Box(low=0, high=1, shape=(1, 4410)), ## edit range and shape later
                # "active_acous_fft": gym.spaces.Box(low=0, high=high, shape=(1, 2206)),
                # "active_acous_spec": gym.spaces.Box(low=0, high=high, shape=(129, 65, 1)),
                # "active_acous_spec": gym.spaces.Box(low=0, high=high, shape=(40, 65, 1)), # if spec is cropped to a specific range
                # "tactile_image": gym.spaces.Box(low=0, high=255, shape=(320, 240, 3)),
                "force_array": gym.spaces.Box(low=0, high=high, shape=(4, 100, 1)),
            }
        )

    def _get_observation(self):
        """If successful, returns (obs, True); otherwise, returns (None, False)."""
        robot_state, panda_error = self.robot.get_state()
        # _, _, image1, _, image2, _, _, _, _, _, active_acous_spec = self.furniture.get_parts_poses()
        # _, _, image1, _, image2, _, _, _, _, active_acous_fft, _ = self.furniture.get_parts_poses()
        # _, _, image1, _, image2, _, _, _, tactile_image = self.furniture.get_parts_poses()
        _, _, image1, _, image2, _, _, _, force_array = self.furniture.get_parts_poses()

        image1 = resize(image1)
        image2 = resize_crop(image2)

        # crop active acous spec to [3000, 10000] Hz
        # active_acous_spec = active_acous_spec[18:58]

        # tactile image process
        # tactile_image = tactile_image

        # force array process
        force_array = force_array

        return (
            # dict(robot_state.__dict__, color_image1=image1, color_image2=image2, active_acous=active_acous),
            dict(
                robot_state.__dict__, 
                color_image1=image1, 
                color_image2=image2, 
                # active_acous_spec=active_acous_spec
                # active_acous_fft=active_acous_fft
                # tactile_image=tactile_image,
                force_array=force_array,
                ), # key name matters
            panda_error,
        )

    @set_frequency(config["robot"]["hz"])
    def step(self, action):
        """Robot takes an action.

        Args:
            action:
                np.ndarray of size 8 (dx, dy, dz, x, y, z, w, grip)
        """
        obs, obs_error = self._get_observation()
        
        if self.act_rot_repr == "rot_6d":
            import pytorch3d.transforms as pt

            rot_6d = action[3:9]
            rot_6d = torch.from_numpy(rot_6d)
            rot_mat = pt.rotation_6d_to_matrix(rot_6d)
            quat = pt.matrix_to_quaternion(rot_mat)
            quat = quat.numpy()
            action = np.concatenate([action[:3], quat, action[-1:]], axis=0)

        action_success = self.robot.execute(action)

        if obs_error != PandaError.OK:
            return None, 0, True, {"obs_success": False, "error": obs_error}

        self.env_steps += 1

        done = self._done()
        if self.manual_reset:
            _, collect_enum = self.device_interface.get_action()
            if collect_enum == CollectEnum.RESET:
                done = True
        
        # add batch dimension to everything.
        # for k in obs:
        #     obs[k] = np.expand_dims(obs[k], axis=0)
        reward = self._reward()
        # reward = np.expand_dims(reward, axis=0)
        # done = np.expand_dims(done, axis=0)
        
                
        return obs, reward, done, {"action_success": action_success}
