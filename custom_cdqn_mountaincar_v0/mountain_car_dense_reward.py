import cv2
import gym
import numpy as np


class MountainCarDenseReward(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, temporal_length=3, debug=0):
        super(MountainCarDenseReward, self).__init__()
        self._env = gym.make("MountainCar-v0")
        self.action_space = self._env.action_space
        self.observation_space = self._env.observation_space

        self._temporal_length = temporal_length
        self._debug = debug

        self._sum_reward = 0
        self._steps = 0
        self._location_previous = None

    def step(self, action):
        images = []
        for _ in range(self._temporal_length):
            obs, reward, done, info = self._env.step(action)
            images.append(
                cv2.resize(
                    cv2.cvtColor(
                        self._env.render(mode="rgb_array"), cv2.COLOR_BGR2GRAY
                    ), (224, 224)
                ).reshape((224, 224))
            )
            if done:
                if len(images) < self._temporal_length:
                    images = images + \
                        [images[-1]] * (self._temporal_length - len(images))
                break
        images = np.expand_dims(np.array(images) / 255, axis=0)
        
        if self._debug > 2:
            print("Image:", images.shape)

        if obs[0] >= .5:
            reward = 200 - self._steps

            if self._debug > 0:
                print("**********solved**********")
        elif self._location_previous is not None:
            reward = abs(obs[0] + .5)

        self._location_previous = obs[0]

        self.steps += 1
        self._sum_reward += reward

        return images, reward, done, info
    
    def reset(self):
        if self._debug > 1:
            print("Steps: {} reward: {:.3f}".format(
                self._steps, self._sum_reward))
        
        self.steps = 0
        self._sum_reward = 0

        self._env.reset()
        
        images = []
        for _ in range(self._temporal_length):
            images.append(
                cv2.resize(
                    cv2.cvtColor(
                        self._env.render(mode="rgb_array"), cv2.COLOR_BGR2GRAY
                    ), (224, 224)
                ).reshape((224, 224))
            )
            self._env.step(1)
        images = np.expand_dims(np.array(images) / 255, axis=0)

        return images
    
    def render(self, mode='human'):
        return self._env.render(mode=mode)
    
    def close (self):
        self._env.close()


if __name__ == "__main__":
    env = MountainCarDenseReward(debug=True)

    env.reset()
    env.step(0)
