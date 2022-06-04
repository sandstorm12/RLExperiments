FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

RUN apt update -y \
    && apt install -y xvfb \
    python-opengl

WORKDIR /workdir

ADD ./requirements.txt .
RUN pip install -r ./requirements.txt

ADD . .
