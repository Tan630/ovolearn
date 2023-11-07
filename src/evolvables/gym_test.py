import gymnasium as gym

visible = False
if visible:
    env = gym.make("CartPole-v1", render_mode="human")
else:
    env = gym.make("CartPole-v1")
env.reset()
score = 0.

for i in range(0,100):
    step_result = env.step(1)
    if (step_result[2]):
        break
    score = score + step_result[1]# type:ignore

print (score)
# print(score)