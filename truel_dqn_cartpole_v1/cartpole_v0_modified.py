import gym


class CartpoleNegativeEnding(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, debug=False):
        super(CartpoleNegativeEnding, self).__init__()
        self._env = gym.make("CartPole-v1")
        self.action_space = self._env.action_space
        self.observation_space = self._env.observation_space

        self._debug = debug

        self._sum_reward = 0
        self._steps = 0

    def step(self, action):
        obs, reward, done, info = self._env.step(action)

        if done:
            reward = -30

        self.steps += 1
        self._sum_reward += reward

        return obs, reward, done, info
    
    def reset(self):
        if self._debug:
            print("Steps: {} reward: {:.3f}".format(self.steps, self._sum_reward))
        
        self.steps = 0
        self._sum_reward = 0

        return self._env.reset()
    
    def render(self, mode='human'):
        return self._env.render(mode=mode)
    
    def close (self):
        self._env.close()
