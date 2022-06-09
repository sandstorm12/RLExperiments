from dqn import DQN
from lunar_lander import LunarLander


def run():
    env = LunarLander(debug=1)

    dqn = DQN(env)
    dqn.learn(200000)

    dqn.save("./model.pth")
    dqn.load("./model.pth")

    dqn.evaluate(capture_path="./output.mp4", episodes=10)


if __name__ == "__main__":
    run()
