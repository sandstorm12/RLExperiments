import pyvirtualdisplay

from dqn import DQN
from lunar_lander import LunarLander


def run():
    pyvirtualdisplay.Display(visible=0, size=(1280, 720)).start()

    env = LunarLander(debug=1)

    dqn = DQN(env)
    dqn.learn(100000)

    dqn.save("./model.pth")
    dqn.load("./model.pth")

    dqn.evaluate(capture_path="./output/output.mp4", episodes=10)


if __name__ == "__main__":
    run()
