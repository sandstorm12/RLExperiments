import gym
import numpy as np
import pyvirtualdisplay

from stable_baselines import A2C
from stable_baselines.common import make_vec_env
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env.vec_video_recorder import VecVideoRecorder


def _initialize_env(num_envs=4):
    env = make_vec_env("CartPole-v1", n_envs=num_envs)

    return env

def _train_model(env):
    model = A2C(MlpPolicy, env, verbose=1)
    model.learn(total_timesteps=250000)

    return model
    
def _test_model(model, env, length=16):
    pyvirtualdisplay.Display(visible=0, size=(1920, 1080)).start()

    done = 0
    env = VecVideoRecorder(
        env, "output/", lambda x: True, video_length=500
    )

    steps = np.zeros(4)
    while True:
        action, _states = model.predict(
            obs if "obs" in locals() else env.reset()
        )
        obs, rewards, dones, info = env.step(action)
        env.render("rgb_array")

        steps += 1
        if np.any(dones):
            done += 1
            print("Done --> steps: {}".format(steps[dones]))
            steps[dones] = 0
            if done >= length:
                break


if __name__ == "__main__":
    env = _initialize_env()

    model = _train_model(env)

    # To demonstrate saving and loading
    model.save("a2c_cartpole_v1")
    model = A2C.load("a2c_cartpole_v1")

    _test_model(model, env)

    env.close()
