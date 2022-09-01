from dqn import DQN
from cartpole_v0_modified import CartpoleNegativeEnding


def run():
    env = CartpoleNegativeEnding()

    dqn = DQN(env)
    dqn.learn(20000)

    dqn.save("./model.pth")
    dqn.load("./model.pth")

    dqn.evaluate(capture_path="./output.mp4", episodes=10)


if __name__ == "__main__":
    run()
