FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04
ENV PYTHON_VERSION=3.6
ENV DEBIAN_FRONTEND=noninteractive
ENV FORCE_CUDA="1"
ENV ROOT=/root/deepsnake
ENV CUDA_HOME=/usr/local/cuda-10.0
ENV NVIDIA_VISIBLE_DEVICES=all
ENV TORCH_CUDA_ARCH_LIST="Turing"
# for use on Turing arch such as Geforce 2070, 2080
ENV CUDA_LAUNCH_BLOCKING=1

RUN ln -fs /usr/share/zoneinfo/America/Los_Angeles /etc/localtime

RUN apt-get update && apt-get install -y --no-install-recommends --fix-missing \
    tzdata \
    git \
    curl \
    vim \
    unzip \
    openssh-client \
    wget \
    build-essential \
    cmake \
    checkinstall \
    gcc \
    tmux \
    libgtk2.0-dev \
    python3.6-distutils  \
    python3.6-dev

RUN ln -sf /usr/bin/python3.6 /usr/bin/python

RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python3 get-pip.py && \
    rm get-pip.py

RUN apt-get install -y --no-install-recommends \
    build-essential \
    libssl-dev \
    libffi-dev \
    libxml2-dev \
    libxslt1-dev \
    zlib1g-dev \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ffmpeg

RUN pip3 --no-cache-dir install \
    numpy \
    scipy \
    sklearn \
    scikit-image \
    pandas \
    matplotlib \
    Cython \
    opencv-python \
    jupyter \
    termcolor \
    requests \
    imgaug \
    tqdm \
    pycocotools \
    pyyaml \
    cupy-cuda100 \
    wheel

WORKDIR /root/deepsnake/

COPY requirements.txt /tmp/requirements.txt

#copy all the files to the container
COPY . .

RUN echo $(python -V) && echo $(which python) && echo "$(nvcc --version)"

RUN pip3 install -U https://download.pytorch.org/whl/cu100/torch-1.0.0-cp36-cp36m-linux_x86_64.whl
RUN git clone https://github.com/DesperateMaker/apex.git /usr/src/apex && cd /usr/src/apex/ && \
    python3 setup.py install --cuda_ext --cpp_ext

RUN python collect_env.py

RUN python3 -c "import torch; print('Torch available: {}'.format(torch.cuda.is_available()))"

WORKDIR ${ROOT}/lib/csrc

# Do this at runtime
RUN cd dcn_v2 && python3 setup.py build_ext --inplace
RUN cd extreme_utils/   && python3 setup.py build_ext --inplace
RUN cd roi_align_layer/ && python3 setup.py build_ext --inplace

WORKDIR /root/deepsnake/
