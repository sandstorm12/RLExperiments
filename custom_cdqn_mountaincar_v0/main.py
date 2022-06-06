import pyvirtualdisplay

from dqn import DQN
from mountain_car_dense_reward import MountainCarDenseReward


def run():
    pyvirtualdisplay.Display(visible=0, size=(1280, 720)).start()

    env = MountainCarDenseReward(debug=1)

    dqn = DQN(env)
    dqn.learn(50000)

    dqn.save("./model.pth")
    dqn.load("./model.pth")

    dqn.evaluate(capture_path="./output/output.mp4", episodes=10)


if __name__ == "__main__":
    run()
