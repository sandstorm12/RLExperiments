import copy
import collections
import numpy as np

from gym import Env
from QNetwork import QNetwork


class DQN:
    def __init__(self, env: Env, memory_size=1000):
        self.network_main = self._initialize_network(env)
        self.network_target = copy.deepcopy(self.network_main)
        
        self.memory_size = memory_size
        self.memory = self._initialize_memory(self.memory_size)

        self.previous_state = None

    def _initialize_memory(memory_size):
        memory = collections.deque(maxlen=memory_size)
        
        return memory

    def _initialize_network(self, env):
        num_inputs = env.observation_space.shape[0]
        num_outputs = env.action_space.n

        network = DQN.QNetwork(num_inputs, num_outputs)

        return network

    def learn(self, env: Env, num_iterations, batch_size=64,
            novel_interactions=16):
        for iteration in range(num_iterations):
            epsilon = 1. - iteration / num_iterations
            samples = self._sample(batch_size, novel_interactions, epsilon)

    def _sample(self, env: Env, num_samples, novel_interactions, epsilon):
        replay_samples = self._sample_from_memory(
            num_samples - novel_interactions
        )

        novel_samples = self._sample_from_env(
            env, novel_interactions, epsilon
        )

        self._memorize_samples(novel_samples)

        return replay_samples + novel_samples

    def _sample_from_memory(self, num_samples):
        replay_samples = [
            self.memory[i]
            for i in np.random.choice(
                len(self.memory),
                min(
                    len(self.memory),
                    num_samples
                )
            )
        ]

        return replay_samples

    def _sample_from_env(self, env, num_samples, epsilon):
        novel_samples = []
        while len(novel_samples) < num_samples:
            if self.previous_state is None:
                self.previous_state = env.reset()

            if np.random.rand() < epsilon:
                action = np.random.randint(0, env.action_space.n)
            else:
                # Select action using the main model
                pass

            state_new, reward, done, info = \
                env.step(action)

            novel_samples.append(
                (self.previous_state.copy(), action, reward, state_new)
            )

            if done:
                self.previous_state = None

        return novel_samples

    def _memorize_samples(self, samples):
        for sample in samples:
            if len(self.memory) == self.memory_size:
                self.memory.popleft()
            
            self.memory.append(sample)

    def predict(self, observation):
        pass
