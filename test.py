# imports
from time import sleep
import gymnasium
import gait_40_dof_22_musc_sagittal as _

# constants
N_RESETS = 10
SLEEP_SECONDS = 0.01

# init
env = gymnasium.make("gait_40_dof_22_musc_sagittal_vel_reward-v0")
env.reset()

# loop
print(f"Running {N_RESETS} episodes..")
resets = 0
length = 0
while True:
    # render
    env.mj_render()

    # step
    a = env.action_space.sample()
    stuff = env.step(a)

    #reset
    if stuff[2]:
        print(f"Episode {resets} completed after {length} steps.")
        resets += 1
        length = 0
        env.reset()
        if resets >= N_RESETS:
            break

    #sleep
    length += 1
    sleep(SLEEP_SECONDS)

print("Test successful!")
