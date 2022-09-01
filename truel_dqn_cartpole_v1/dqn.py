import os
import time
import copy
import torch
import collections
import numpy as np
import pyvirtualdisplay
import matplotlib.pyplot as plt

from tqdm import tqdm

from q_network import QNetwork

from gym import Env
from gym.wrappers.monitoring import video_recorder


class DQN:
    def __init__(self, env: Env, memory_size=1000, discount_factor=.99,
            target_network_learning_rate=1e-3):
        self._env = env

        self._network_main = self._initialize_network(self._env)

        self._memory_size = memory_size
        self._discount_factor = discount_factor
        self._target_network_learning_rate = target_network_learning_rate

        self._dump_dir = f"./exp_{time.time()}"
        os.makedirs(self._dump_dir)

    def _initialize_training(self, ):
        self._network_target = copy.deepcopy(self._network_main)
        self._network_target_mid = copy.deepcopy(self._network_main)
        
        self._memory = self._initialize_memory(self._memory_size)

        self._epsilon = None
        self._previous_state = None
        self._average_reward = 0

        self._avg_step_rewards = []
        self._avg_episode_rewards = []
        self._var_step_rewards = []
        self._var_episode_rewards = []
        self._ma_step_rewards = []
        self._ma_episode_rewards = []
        self._grad_step_rewards = []
        self._grad_episode_rewards = []

    def _initialize_memory(self, memory_size):
        memory = collections.deque(maxlen=memory_size)
        
        return memory

    def _initialize_network(self, env):
        num_inputs = env.observation_space.shape[0]
        num_outputs = env.action_space.n

        network = QNetwork(num_inputs, num_outputs)

        return network

    def _uninitialize_training(self):
        del self._memory
        del self._network_target
        del self._network_target_mid
        del self._epsilon
        del self._previous_state
        del self._average_reward
        del self._avg_step_rewards
        del self._avg_episode_rewards
        del self._var_step_rewards
        del self._var_episode_rewards
        del self._ma_step_rewards
        del self._ma_episode_rewards
        del self._grad_step_rewards
        del self._grad_episode_rewards

    def learn(self, num_iterations, batch_size=64,
            novel_interactions=16):
        self._initialize_training()

        bar = tqdm(range(num_iterations))
        for iteration in bar:
            self._epsilon = 1. - iteration / num_iterations
            samples = self._sample(batch_size, novel_interactions)

            self._train_networks(samples)
            self._report(iteration, samples, bar)
            self._sync_networks()

            if iteration % 100 == 99:
                self._measure_volatility()

            if iteration % 1000 == 999:
                self.evaluate(episodes=10)

        self._uninitialize_training()

    def _report(self, iteration, samples, bar, period=1000):
        if iteration % period == 0:
            bar.set_description(
                "Epsilon: {:.3f} reward: {:.3f}".format(
                    self._epsilon, self._average_reward
                )
            )

            self._average_reward = 0
        else:
            weight = (iteration % period)
            samples_average_reward = np.mean(
                [sample[2] for sample in samples])
            self._average_reward = (
                self._average_reward * weight + samples_average_reward
            ) / (weight + 1)

    def _sync_networks(self):
        for parameter_1, parameter_2 in zip(
                self._network_main.parameters(),
                self._network_target.parameters()):
            parameter_2.data.copy_(
                self._target_network_learning_rate * parameter_1 +
                (1 - self._target_network_learning_rate) * parameter_2
            )

        for parameter_1, parameter_2 in zip(
                self._network_target.parameters(),
                self._network_target_mid.parameters()):
            parameter_2.data.copy_(
                self._target_network_learning_rate * parameter_1 +
                (1 - self._target_network_learning_rate) * parameter_2
            )

    def _train_networks(self, samples):
        rewards = torch.tensor([sample[2] for sample in samples])
        actions = torch.tensor([sample[1] for sample in samples])
        states = torch.tensor(
            np.array([sample[0] for sample in samples])).float()
        states_next = torch.tensor(
            np.array([sample[3] for sample in samples])).float()

        q_current = self._network_main(states).gather(
            1, actions.view(-1, 1)).view(-1)
        
        q_target = self._network_target(states_next)
        q_target_mid = self._network_target_mid(states_next)
        q_best = (rewards + \
            self._discount_factor * torch.max(q_target, dim=1)[0]).float()
        q_best_mid = (rewards + \
            self._discount_factor * torch.max(q_target_mid, dim=1)[0]).float()
        

        if len(self._grad_episode_rewards) > 0:
            coeff = min(10, max(0, self._grad_episode_rewards[-1])) / 10
            q_best_total = coeff * q_best + (1 - coeff) * q_best_mid

            self._network_main.train(q_current.float(), q_best_total.float())

    def _sample(self, num_samples, novel_interactions):
        replay_samples = self._sample_from_memory(
            num_samples - novel_interactions
        )

        novel_samples = self._sample_from_env(
            min(novel_interactions, num_samples - len(replay_samples))
        )

        self._memorize_samples(novel_samples)

        return replay_samples + novel_samples

    def _sample_from_memory(self, num_samples):
        replay_samples = [
            self._memory[i]
            for i in np.random.choice(
                len(self._memory),
                min(
                    len(self._memory),
                    num_samples
                )
            )
        ]

        return replay_samples

    def _sample_from_env(self, num_samples):
        novel_samples = []
        while len(novel_samples) < num_samples:
            if self._previous_state is None:
                self._previous_state = self._env.reset()

            if np.random.rand() < self._epsilon:
                action = np.random.randint(0, self._env.action_space.n)
            else:
                previous_state_tensor = torch.from_numpy(
                    self._previous_state).unsqueeze(0).float()
                action = torch.argmax(
                    self._network_main(previous_state_tensor), dim=1).item()

            state_new, reward, done, _ = \
                self._env.step(action)

            novel_samples.append(
                (self._previous_state.copy(), action, reward, state_new, done)
            )

            if done:
                self._previous_state = None
            else:
                self._previous_state = state_new

        return novel_samples

    def _memorize_samples(self, samples):
        for sample in samples:
            for _ in range(int(np.sqrt(max(1, sample[2])))):
                if len(self._memory) == self._memory_size:
                    del self._memory[np.random.randint(0, self._memory_size)]
                
                self._memory.append(sample)

    def save(self, path):
        torch.save(self._network_main, path)

    def load(self, path):
        self._network_main = torch.load(path)

    def evaluate(self, capture_path=None, episodes=1):
        pyvirtualdisplay.Display(visible=0, size=(1920, 1080)).start()
        recorder = video_recorder.VideoRecorder(
            self._env, capture_path, enabled=capture_path is not None)
        for _ in range(episodes):
            steps = 0
            rewards = []
            while True:
                action = self.predict(
                    obs if "obs" in locals() else self._env.reset()
                )
                obs, reward, done, info = self._env.step(action)
                recorder.capture_frame()

                steps += 1
                rewards.append(reward)

                if done:
                    print(
                        "Done --> steps {} reward: {:.3f} {:.3f}".format(
                            steps, np.mean(rewards), np.sum(rewards)
                        )
                    )
                    obs = self._env.reset()
                    break

        recorder.close()
        recorder.enabled = False
                    

    def predict(self, observation):
        observation = torch.from_numpy(observation).unsqueeze(0).float()
        action = torch.argmax(self._network_main(observation), dim=1).item()

        return action

    def _measure_volatility(self, episodes=10):
        rewards_episode = []
        rewards_steps = []
        for _ in range(episodes):
            steps = 0
            rewards = []
            while True:
                action = self.predict(
                    obs if "obs" in locals() else self._env.reset()
                )
                obs, reward, done, info = self._env.step(action)

                steps += 1
                rewards.append(reward)

                if done:
                    obs = self._env.reset()
                    break

            rewards_episode.append(np.sum(rewards))
            rewards_steps.extend(rewards)

        self._avg_step_rewards.append(np.mean(rewards_steps))
        self._avg_episode_rewards.append(np.mean(rewards_episode))
        self._var_step_rewards.append(np.var(rewards_steps))
        self._var_episode_rewards.append(np.var(rewards_episode))

        if len(self._ma_step_rewards) != 0:
            self._ma_step_rewards.append(
                self._ma_step_rewards[-1] * .9 + self._avg_step_rewards[-1] * .1)
            self._ma_episode_rewards.append(
                self._ma_episode_rewards[-1] * .9 + self._avg_episode_rewards[-1] * .1)
        else:
            self._ma_step_rewards.append(self._avg_step_rewards[-1])
            self._ma_episode_rewards.append(self._avg_episode_rewards[-1])

        if len(self._ma_step_rewards) > 10:
            self._grad_step_rewards.append(np.mean(np.gradient(self._ma_step_rewards[-50:])))
            self._grad_episode_rewards.append(np.mean(np.gradient(self._ma_episode_rewards[-10:])))
        else:
            self._grad_step_rewards.append(0)
            self._grad_episode_rewards.append(0)

        self._plot_volatility()

    def _plot_volatility(self):
        plt.plot(self._avg_step_rewards)
        plt.savefig(
            os.path.join(
                self._dump_dir, f"{time.time()}_avg_step_rewards.png"))
        plt.close()

        plt.plot(self._avg_episode_rewards)
        plt.savefig(
            os.path.join(
                self._dump_dir, f"{time.time()}_avg_episode_rewards.png"))
        plt.close()

        plt.plot(self._var_step_rewards)
        plt.savefig(
            os.path.join(
                self._dump_dir, f"{time.time()}_var_step_rewards.png"))
        plt.close()

        plt.plot(self._var_episode_rewards)
        plt.savefig(
            os.path.join(
                self._dump_dir, f"{time.time()}_var_episode_rewards.png"))
        plt.close()

        plt.plot(self._ma_step_rewards)
        plt.savefig(
            os.path.join(
                self._dump_dir, f"{time.time()}_ma_step_rewards.png"))
        plt.close()

        plt.plot(self._ma_episode_rewards)
        plt.savefig(
            os.path.join(
                self._dump_dir, f"{time.time()}_ma_episode_rewards.png"))
        plt.close()
        
        plt.plot(self._grad_step_rewards)
        plt.savefig(
            os.path.join(
                self._dump_dir, f"{time.time()}_grad_step_rewards.png"))
        plt.close()

        plt.plot(self._grad_episode_rewards)
        plt.savefig(
            os.path.join(
                self._dump_dir, f"{time.time()}_grad_episode_rewards.png"))
        plt.close()
