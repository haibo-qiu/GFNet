#
# This example Dockerfile illustrates a method to install
# additional packages on top of NVIDIA's PyTorch container image.
#
# To use this Dockerfile, use the `docker build` command.
# See https://docs.docker.com/engine/reference/builder/
# for more information.
#
FROM nvcr.io/nvidia/pytorch:20.12-py3

RUN apt-get update && apt-get install -y python3-opencv\
      && rm -rf /var/lib/apt/lists/

RUN pip install opencv-contrib-python==4.4.0.46 torch_scatter==2.0.8 && \
    pip install dropblock==0.3.0 tensorboardX==2.0 && \
    pip install prettytable plyfile timm nuscenes-devkit
