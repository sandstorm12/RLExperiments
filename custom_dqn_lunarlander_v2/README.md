# MountainCar v0

Solving the `LunarLander-v2` using a custom `DQN` agent and parameteric observations.

![Alt Text](data/output.gif)

## Notice

The requirements are tested using `Python36`


## Requirements


```bash
apt install -y xvfb python-opengl

pip install --upgrade pip
pip install -r requirements.txt
```

OR

```bash
docker build -t custom_dqn_lunarlander_v2 .
docker run -it --rm -v $(pwd):/workdir custom_dqn_lunarlander_v2 bash
```

## Run

```bash
python main.py
```

**The output videos will be stored in the output directory*


## Urgent issues and future work
1. *Nothing so far*


## Issues and future work
1. Bug: Results are not perfect, more training, better model, or a change in the environment is required


## Contributors

1. Hamid Mohammadi: <sandstormeatwo@gmail.com>
