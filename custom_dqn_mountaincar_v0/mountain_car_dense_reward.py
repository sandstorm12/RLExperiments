import gym


class MountainCarDenseReward(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, debug=False):
        super(MountainCarDenseReward, self).__init__()
        self.env = gym.make("MountainCar-v0")
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

        self.debug = debug

        self.sum_reward = 0
        self.steps = 0
        self.location_previous = None

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        if obs[0] >= .5:
            reward = 200 - self.steps

            if self.debug:
                print("**********solved**********")
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
