import os
from typing import Dict, Tuple, Union

import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 1,
    "distance": 1.5,
    "lookat": np.array((0.0, 0.0, 0.5)),
    "elevation": -5.0,
}

def mass_center(model, data):
    mass = np.expand_dims(model.body_mass, axis=1)
    xpos = data.xipos
    return (np.sum(mass * xpos, axis=0) / np.sum(mass))[0:2].copy()

# you can completely modify this class for your MuJoCo environment by following the directions
class RobotisEnv(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 100,
    }

    # set default episode_len for truncate episodes
    def __init__(
        self,         
        frame_skip: int = 5,
        default_camera_config: Dict[str, Union[float, int]] = DEFAULT_CAMERA_CONFIG,
        forward_reward_weight: float = 5.00,
        ctrl_cost_weight: float = 0.001,
        ctrl_cost_diff_axis_y: float = 0.1,
        standing_cost: float = 0.1,
        healthy_reward: float = 2.0,
        terminate_when_unhealthy: bool = True,
        healthy_z_range: Tuple[float, float] = (0.260, 0.290),
        reset_noise_scale: float = 1e-2,
        # exclude_current_positions_from_observation: bool = True,
        include_cinert_in_observation: bool = False,
        include_cvel_in_observation: bool = False,
        include_qfrc_actuator_in_observation: bool = False,
        # include_cfrc_ext_in_observation: bool = True, 
        **kwargs):

        utils.EzPickle.__init__(
            self, 
            frame_skip,
            default_camera_config,
            forward_reward_weight,
            ctrl_cost_weight,
            ctrl_cost_diff_axis_y,
            # contact_cost_weight,
            # contact_cost_range,
            standing_cost,
            healthy_reward,
            terminate_when_unhealthy,
            healthy_z_range,
            reset_noise_scale,
            include_cinert_in_observation,
            include_cvel_in_observation,
            include_qfrc_actuator_in_observation,
            **kwargs
        )
        self._forward_reward_weight: float = forward_reward_weight
        self._ctrl_cost_weight: float = ctrl_cost_weight
        self._ctrl_cost_diff_axis_y: float = ctrl_cost_diff_axis_y
        self._standing_cost: float = standing_cost
        self._healthy_reward: float = healthy_reward
        self._terminate_when_unhealthy: bool = terminate_when_unhealthy
        self._healthy_z_range: Tuple[float, float] = healthy_z_range
        self._reset_noise_scale: float = reset_noise_scale

        self._include_cinert_in_observation = include_cinert_in_observation
        self._include_cvel_in_observation = include_cvel_in_observation
        self._include_qfrc_actuator_in_observation = include_qfrc_actuator_in_observation

        MujocoEnv.__init__(
            self,
            os.path.join(os.path.dirname(__file__), "robotis_mjcf", "scene.xml"),
            frame_skip,
            observation_space=None,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs
        )

        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
            ],
            "render_fps": int(np.round(1.0 / self.dt)),
        }

        obs_size = self.data.qpos.size + self.data.qvel.size + self.data.sensordata.size
        obs_size += self.data.cinert[1:].size * include_cinert_in_observation
        obs_size += self.data.cvel[1:].size * include_cvel_in_observation
        obs_size += (self.data.qvel.size - 6) * include_qfrc_actuator_in_observation
        # obs_size += self.data.cfrc_ext[1:].size * include_cfrc_ext_in_observation

        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64
        )


    # determine the reward depending on observation or other properties of the simulation
    def step(self, action):
        xy_position_before = mass_center(self.model, self.data)
        self.do_simulation(action, self.frame_skip)
        xy_position_after = mass_center(self.model, self.data)

        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity

        x_pos_delta = xy_position_after[0] - xy_position_before[0]

        observation = self._get_obs()
        # reward, reward_info = self._get_rew(x_velocity, x_pos_delta, action)
        reward, reward_info = self._get_rew(x_velocity, self.data.qpos[0], x_pos_delta, action)
        terminated = (not self.is_healthy) # and self._terminate_when_unhealthy
        info = {
            "x_position": self.data.qpos[0],
            "y_position": self.data.qpos[1],
            "distance_from_origin": np.linalg.norm(self.data.qpos[0:2], ord=2),
            "x_velocity": x_velocity,
            "y_velocity": y_velocity,
            "z_height": self.data.site('torso').xpos[2],
            # "z_height": self.data.qpos[2],
            "x_pos_delta": x_pos_delta,
            **reward_info,
        }

        if self.render_mode == "human":
            self.render()
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return observation, reward, terminated, False, info


    @property
    def is_healthy(self):
        min_z, max_z = self._healthy_z_range
        # is_healthy = min_z < self.data.site('torso').xpos[2] < max_z
        is_healthy = min_z < self.data.qpos[2] < max_z
        return is_healthy

    @property
    def healthy_reward(self):
        return self.is_healthy * self._healthy_reward      

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(self.data.ctrl))
        return control_cost

    def _get_rew(self, x_velocity: float, x_pos:float, pos_delta:float, action):
        forward_reward = (self._forward_reward_weight * x_pos)
        healthy_reward = self.healthy_reward
        rewards = forward_reward + healthy_reward 

        ctrl_cost = self.control_cost(action)
        diff_y_axis = abs(self.data.site('torso').xpos[1]) * self._ctrl_cost_diff_axis_y
        penalty_reward = (pos_delta < 0.01) * self._standing_cost
        costs = ctrl_cost + diff_y_axis + penalty_reward

        reward = rewards - costs

        reward_info = {
            "reward_survive": healthy_reward,
            "reward_forward": forward_reward,
            "reward_ctrl": -ctrl_cost,
            "reward_diff_y_axis": -diff_y_axis,
        }

        return reward, reward_info

    # define what should happen when the model is reset (at the beginning of each episode)
    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nv
        )
        self.set_state(qpos, qvel)

        return self._get_obs()

    # determine what should be added to the observation
    # for example, the velocities and positions of various joints can be obtained through their names, as stated here
    def _get_obs(self):
        position = self.data.qpos.flatten()
        velocity = self.data.qvel.flatten()
        imu = self.data.sensordata.flatten()
        # com_inertia = self.data.cinert[1:].flatten()
        # com_velocity = self.data.cvel[1:].flatten()
        # actuator_forces = self.data.qfrc_actuator[6:].flatten()

        if self._include_cinert_in_observation is True:
            com_inertia = self.data.cinert[1:].flatten()
        else:
            com_inertia = np.array([])
        
        if self._include_cvel_in_observation is True:
            com_velocity = self.data.cvel[1:].flatten()
        else:
            com_velocity = np.array([])

        if self._include_qfrc_actuator_in_observation is True:
            actuator_forces = self.data.qfrc_actuator[6:].flatten()
        else:
            actuator_forces = np.array([])
        

        return np.concatenate(
            (
                position,
                velocity,
                imu,
                com_inertia,
                com_velocity,
                actuator_forces,
            )
        )
    
    def _get_reset_info(self):
        return {
            "x_position": self.data.qpos[0],
            "y_position": self.data.qpos[1],
            "distance_from_origin": np.linalg.norm(self.data.qpos[0:2], ord=2),
        }        