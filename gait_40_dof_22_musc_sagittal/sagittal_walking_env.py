import gym
import numpy as np
from myosuite.envs.myo.myobase.walk_v0 import WalkEnvV0
from myosuite.envs.myo.base_v0 import BaseV0
import collections


class WalkingSagittalLeft(WalkEnvV0):
    DEFAULT_OBS_KEYS = [
        'qpos_without_xy',
        'qvel',
        'com_vel',
        'torso_angle',
        'feet_heights',
        'height',
        'feet_rel_positions',
        'phase_var',
        'muscle_length',
        'muscle_velocity',
        'muscle_force'
    ]
    DEFAULT_RWD_KEYS_AND_WEIGHTS = {
        "vel_reward": 5.0,
        "done": -100,
        "cyclic_hip": -10,
        "ref_rot": 10.0,
    }

    def __init__(self, model_path, obsd_model_path=None, 
                 seed=None, **kwargs):
        gym.utils.EzPickle.__init__(self, 
                                    model_path,
                                    obsd_model_path, 
                                    seed, 
                                    **kwargs)
        BaseV0.__init__(self, 
                        model_path=model_path, 
                        obsd_model_path=obsd_model_path, 
                        seed=seed, 
                        env_credits=self.MYO_CREDIT)

        self.first_done_call = True
        self.max_act_mag = kwargs["max_act_mag"] \
            if "max_act_mag" in kwargs.keys() \
            else 0.1

        self._setup(**kwargs)

    def reset(self):
        self.steps = 0
        reset_qpos = self.sim.model.key_qpos[0]
        reset_qvel = self.sim.model.key_qvel[0]

        self.robot.sync_sims(self.sim, self.sim_obsd)
        obs = BaseV0.reset(self, 
                           reset_qpos=reset_qpos, 
                           reset_qvel=reset_qvel)
        return obs

    def get_obs_dict(self, sim):
         obs_dict = {}
         obs_dict['t'] = np.array([sim.data.time])
         obs_dict['time'] = np.array([sim.data.time])
         obs_dict['qpos_without_xy'] = sim.data.qpos[1:].copy()
         obs_dict['qvel'] = sim.data.qvel[:].copy() * self.dt
         obs_dict['com_vel'] = np.array([self._get_com_velocity().copy()])
         obs_dict['torso_angle'] = np.array([self._get_torso_angle().copy()])
         obs_dict['feet_heights'] = self._get_feet_heights().copy()
         obs_dict['height'] = np.array([self._get_height()]).copy()
         obs_dict['feet_rel_positions'] \
            = self._get_feet_relative_position().copy()
         obs_dict['phase_var'] \
            = np.array([(self.steps/self.hip_period) % 1]).copy()
         obs_dict['muscle_length'] = self.muscle_lengths()
         obs_dict['muscle_velocity'] = self.muscle_velocities()
         obs_dict['muscle_force'] = self.muscle_forces()

         if sim.model.na>0:
             obs_dict['act'] = sim.data.act[:].copy()

         return obs_dict

    def get_reward_dict(self, obs_dict):
        vel_loss = self._get_vel_loss()
        vel_reward = self._get_vel_reward()
        cyclic_hip = self._get_cyclic_rew()
        ref_rot = self._get_ref_rotation_rew()
        act_mag = np.linalg.norm(
            self.obs_dict['act'], axis=-1
        ) / self.sim.model.na if self.sim.model.na !=0 \
                              else 0
        act_mag = np.array(act_mag).item()
        tollerant_act_mag = np.maximum(
            act_mag - self.max_act_mag, 
            0
        )

        rwd_dict = collections.OrderedDict((
            # Optional Keys
            ('vel_loss', vel_loss),
            ('vel_reward', vel_reward),
            ('cyclic_hip', cyclic_hip),
            ('ref_rot', ref_rot),
            ('act_mag', act_mag),
            ('tollerant_act_mag', tollerant_act_mag),
            # Must keys
            ('sparse', vel_reward),
            ('solved', vel_reward >= 1.0),
            ('done',  self._get_done()),
        ))
        rwd_dict['dense'] = np.sum([wt*rwd_dict[key] 
                                    for key, wt 
                                    in self.rwd_keys_wt.items()],
                                   axis=0)
        return rwd_dict
        
    def _get_vel_loss(self):
        """
        Absolute difference sum as loss 
        """
        vel = self._get_com_velocity()
        return (np.abs(self.target_y_vel - vel[1]) + 
                np.abs(self.target_x_vel - vel[0]))

    def _get_done(self):
        if self.first_done_call:
            self.first_done_call = False
            return 0
        height = self._get_height()
        if height < self.min_height:
            return 1
        return 0

class WalkingSagittalRight(WalkingSagittalLeft):
    def reset(self):
        self.steps = 0
        
        reset_qpos = self.sim.model.key_qpos[1]
        reset_qvel = self.sim.model.key_qvel[1]

        self.robot.sync_sims(self.sim, self.sim_obsd)
        obs = BaseV0.reset(self, reset_qpos=reset_qpos, 
                           reset_qvel=reset_qvel)
        return obs

class WalkingSagittalStochSide(WalkingSagittalLeft):
    def reset(self):
        self.steps = 0
        
        side = np.random.choice([0, 1])
        reset_qpos = self.sim.model.key_qpos[side]
        reset_qvel = self.sim.model.key_qvel[side]

        self.robot.sync_sims(self.sim, self.sim_obsd)
        obs = BaseV0.reset(self, reset_qpos=reset_qpos, 
                           reset_qvel=reset_qvel)
        return obs
        

class WalkingSagittalStochSideNoise(WalkingSagittalLeft):
    def reset(self):
        self.steps = 0
        
        side = np.random.choice([0, 1])
        reset_qpos = self.sim.model.key_qpos[side] 
        reset_qpos += np.random.normal(0, 0.02, 
                                       size=reset_qpos.shape)
        reset_qvel = self.sim.model.key_qvel[side]

        self.robot.sync_sims(self.sim, self.sim_obsd)
        obs = BaseV0.reset(self, reset_qpos=reset_qpos, 
                           reset_qvel=reset_qvel)
        return obs
