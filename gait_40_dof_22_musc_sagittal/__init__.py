import os
from gymnasium import register

curr_dir = os.path.dirname(os.path.abspath(__file__))

register(id='gait_40_dof_22_musc_sagittal-v0',
        entry_point='gait_40_dof_22_musc_sagittal.sagittal_walking_env:'
                    'WalkingSagittalStochSide',
        max_episode_steps=1000,
        kwargs={
            'model_path': curr_dir + '/model/2d.xml',
            'normalize_act': True,
            'min_height':0.6,    # minimum center of mass height before reseSagittal            'hip_period':100,    # desired periodic hip angle movement
            'reset_type':'random', # none, init, random
            'target_x_vel':-1.2,  # desired x velocity in m/s
            'target_y_vel':0.0,  # desired y velocity in m/s
            'weighted_reward_keys': {'act_mag': 0.5}
        }
    )

register(id='gait_40_dof_22_musc_sagittal_noise-v0',
        entry_point='gait_40_dof_22_musc_sagittal.sagittal_walking_env:'
                    'WalkingSagittalStochSideNoise',
        max_episode_steps=1000,
        kwargs={
            'model_path': curr_dir + '/model/2d.xml',
            'normalize_act': True,
            'min_height':0.6,    # minimum center of mass height before reset
            'hip_period':100,    # desired periodic hip angle movement
            'reset_type':'random', # none, init, random
            'target_x_vel':-1.2,  # desired x velocity in m/s
            'target_y_vel':0.0,  # desired y velocity in m/s
            'weighted_reward_keys': {'act_mag': 0.5}
        }
    )

register(id='gait_40_dof_22_musc_sagittal_left-v0',
        entry_point='gait_40_dof_22_musc_sagittal.sagittal_walking_env:'
                    'WalkingSagittalLeft',
        max_episode_steps=1000,
        kwargs={
            'model_path': curr_dir + '/model/2d.xml',
            'normalize_act': True,
            'min_height':0.6,    # minimum center of mass height before reset
            'hip_period':100,    # desired periodic hip angle movement
            'reset_type':'random', # none, init, random
            'target_x_vel':-1.2,  # desired x velocity in m/s
            'target_y_vel':0.0,  # desired y velocity in m/s
            'weighted_reward_keys': {'act_mag': 0.5}
        }
    )

register(id='gait_40_dof_22_musc_sagittal_right-v0',
        entry_point='gait_40_dof_22_musc_sagittal.sagittal_walking_env:'
                    'WalkingSagittalRight',
        max_episode_steps=1000,
        kwargs={
            'model_path': curr_dir + '/model/2d.xml',
            'normalize_act': True,
            'min_height':0.6,    # minimum center of mass height before reset
            'hip_period':100,    # desired periodic hip angle movement
            'reset_type':'random', # none, init, random
            'target_x_vel':-1.2,  # desired x velocity in m/s
            'target_y_vel':0.0,  # desired y velocity in m/s
            'weighted_reward_keys': {'act_mag': 0.5}
        }
    )
