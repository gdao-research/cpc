FROM tensorflow/tensorflow:2.1.0-gpu-py3

# Configure apt and install packages
RUN apt-get update \
    && apt-get -y install htop \
    && apt-get -y install libopenmpi-dev libsm6 libxext6 libxrender-dev \
    && apt-get -y install --no-install-recommends apt-utils dialog 2>&1 \
    && pip install opencv-python\
    #
    # Verify git, process tools, lsb-release (common in install instructions for CLIs) installed
    && apt-get -y install git procps lsb-release \
