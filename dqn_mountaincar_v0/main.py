import gym
import numpy as np
import pyvirtualdisplay

from stable_baselines import DQN
from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.vec_env.vec_video_recorder \
    import VecVideoRecorder


class MountainCarDenseReward(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self):
        super(MountainCarDenseReward, self).__init__()
        self.env = gym.make("MountainCar-v0")
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

        self.sum_reward = 0
        self.steps = 0
        self.location_previous = None

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        if obs[0] >= .5:
            reward = 200 - self.steps
        elif self.location_previous is not None:
            reward = abs(obs[0] + .5)

        self.location_previous = obs[0]

        self.steps += 1
        self.sum_reward += reward

        return obs, reward, done, info
    
    def reset(self):
        print("Steps: {} reward: {:.3f}".format(self.steps, self.sum_reward))
        
        self.steps = 0
        self.sum_reward = 0

        return self.env.reset()
    
    def render(self, mode='human'):
        return self.env.render(mode=mode)
    
    def close (self):
        self.env.close()

make_env_fn = lambda: MountainCarDenseReward()
env = DummyVecEnv([make_env_fn]*1)

pyvirtualdisplay.Display(visible=0, size=(1920, 1080)).start()

model = DQN(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=100000)
model.save("dqn_mountaincar_v0")

model = DQN.load("dqn_mountaincar_v0")

env = VecVideoRecorder(
    env, "output/", lambda x: True, video_length=600
)


for _ in range(10):
    for i in range(200):
        action, _states = model.predict(
            obs if "obs" in locals() else env.reset()
        )
        obs, rewards, dones, info = env.step(action)
        env.render("rgb_array")

        if np.any(dones):
            print("Done --> steps {}".format(i))

env.close()
