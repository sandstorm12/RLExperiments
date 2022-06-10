import gym


class LunarLander(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, debug=1):
        super(LunarLander, self).__init__()
        self._env = gym.make("LunarLander-v2")
        self.action_space = self._env.action_space
        self.observation_space = self._env.observation_space

        self._debug = debug

        self._sum_reward = 0
        self._steps = 0
        self._location_previous = None

    def step(self, action):
        obs, reward, done, info = self._env.step(action)

        self._steps += 1
        self._sum_reward += reward

        if self._debug > 0 and self._sum_reward >= 200:
            print("**********solved**********")

        return obs, reward, done, info
    
    def reset(self):
        if self._debug > 1:
            print("Steps: {} reward: {:.3f}".format(
                    self._steps, self._sum_reward
                )
            )
        
        self.steps = 0
        self._sum_reward = 0

        return self._env.reset()
    
    def render(self, mode='human'):
        return self._env.render(mode=mode)
    
    def close (self):
        self._env.close()
