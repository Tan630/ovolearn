import gymnasium as gym
import numpy as np
env_noviz = gym.make("Pendulum-v1")
env_viz = gym.make("Pendulum-v1", render_mode="human")
env = env_viz # if vizualize else env_noviz

er = np.array([1])
print (er.ndim)
# e = np.ndarray((1,), buffer=np.array([1]))

import random
env.reset()

for i in range(0, 10):
    res = env.step([2])
    res = env.step([2])
    res = env.step([2])
    res = env.step([2])
    res = env.step([2])
    
    res = env.step([-1])
    res = env.step([-1])
    res = env.step([-1])
    res = env.step([-1])
print (res)



def input_wrapper(f: float) -> float:
    return max(min(2, f), -2)
