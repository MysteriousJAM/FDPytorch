FROM nvidia/cuda:11.1-cudnn8-devel-ubuntu20.04
SHELL ["/bin/bash", "-c"]
ENV TERM=linux DEBIAN_FRONTEND=noninteractive
RUN apt-get update -qq -o=Dpkg::Use-Pty=0 && \
    apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-setuptools \
        vim &&\
    python3 -m pip install --upgrade pip
RUN python3 -m pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
RUN python3 -m pip install \
		'easydict'\
		'tqdm' \
		'PyYAML' &&\
	python3 -m pip install --extra-index-url https://developer.download.nvidia.com/compute/redist nvidia-dali-cuda100==0.30.0 &&\
    mkdir /workspace && \
    rm -rf /var/lib/apt/lists/*
WORKDIR workspace/