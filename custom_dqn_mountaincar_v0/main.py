from dqn import DQN
from mountain_car_dense_reward import MountainCarDenseReward


def run():
    env = MountainCarDenseReward(debug=False)

    dqn = DQN(env)
    dqn.learn(100000)

    dqn.save("./model.pth")
    dqn.load("./model.pth")

    dqn.evaluate(capture_path="./output/output.mp4", episodes=10)


if __name__ == "__main__":
    run()
