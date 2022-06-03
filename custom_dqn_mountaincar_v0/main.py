import numpy as np
import pyvirtualdisplay

from mountain_car_custom import MountainCarDenseReward

from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.vec_env.vec_video_recorder \
    import VecVideoRecorder




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
