include ./vnc.config

MJC_VER=131
MJC_DL_URL=https://www.roboti.us/download/mjpro131_linux.zip

step-1: install-conda
step-2: install-python3.5.2
step-3: install get-mujoco install-mujoco-dependencies install-glfw install-mujoco-py install-gym
install-tf-1: tensorflow-dependencies
install-tf-2: tensorflow-dependencies-step-2 build-tensorflow install-tensorflow-wheel
# useful scripts
gpu-monitor:
	watch -n1 "nvidia-smi"
init:
	source ./env
start-display:
	Xvfb :1 -screen 0 1400x900x24 +extension RANDR &
create-vnc-password:
	x11vnc -storepasswd $(VNC_PASSWORD) /tmp/vncpass
start-vnc:
	sudo x11vnc -ncache -rfbport $(VNC_PORT) -rfbauth /tmp/vncpass -display :1 -forever -auth /tmp/xvfb.auth
start-tensorboard:
	tensorboard --logdir=/tmp/tensorflow/

# Installation Scripts
install-conda:
	wget https://repo.continuum.io/archive/Anaconda3-4.4.0-Linux-x86_64.sh
	sudo bash ./Anaconda3-4.4.0-Linux-x86_64.sh -b
	# make sure that we prepend to the bashrc file, because it returns right
	# away if not ran in interactive mode.
	bash ./conda-path.sh
install-python3.5.2:
	bash -c "conda create --name gym python=3.5.2 -y"
install:
	bash -c "source activate gym && pip install -r requirements.txt"

tensorflow-dependencies:
	# Install dependencies
	bash -c "export DEBIAN_FRONTEND=noninteractive && sudo apt-get -yq update"
	bash -c 'sudo DEBIAN_FRONTEND=noninteractive apt-get -y -o Dpkg::Options::="--force-confdef" -o Dpkg::Options::="--force-confold" dist-upgrade'
	sudo apt-get install -y libglu1-mesa libxi-dev libxmu-dev build-essential python-pip python-dev python-numpy python-scipy
	# Java stuff for Google Bazel/Blaze.
	sudo apt-get install -y openjdk-8-jdk git python-dev python3-dev python-numpy python3-numpy build-essential python-pip python3-pip python3-venv swig python3-wheel libcurl3-dev
	# Now install Bazel
	# echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list
	# curl https://storage.googleapis.com/bazel-apt/doc/apt-key.pub.gpg | sudo apt-key add -
	# bash -c "export DEBIAN_FRONTEND=noninteractive && sudo apt-get -yq update"
	# sudo apt-get install -y bazel
	# sudo apt-get upgrade -yq bazel
	# Install Bazel 0.5.2 from binary
	wget https://github.com/bazelbuild/bazel/releases/download/0.5.2/bazel-0.5.2-jdk7-installer-linux-x86_64.sh
	chmod +x bazel-0.5.2-jdk7-installer-linux-x86_64.sh
	./bazel-0.5.2-jdk7-installer-linux-x86_64.sh --user
	bash ./bazel-path.sh
	# kernel sources and compilers
	sudo apt-get install -y gcc g++ gfortran  git linux-image-generic linux-headers-generic linux-source linux-image-extra-virtual libopenblas-dev
	# deactivate neuvu driver
	echo "blacklist nouveau\nblacklist lbm-nouveau\noptions nouveau modeset=0\nalias nouveau off\nalias lbm-nouveau off" | sudo tee /etc/modprobe.d/blacklist-nouveau.conf
	# Now reboot!
	sudo update-initramfs -u
	sudo reboot
tensorflow-dependencies-step-2:
	# Download nvidia driver
	wget -P ~/Downloads/ http://us.download.nvidia.com/XFree86/Linux-x86_64/367.57/NVIDIA-Linux-x86_64-367.57.run
	chmod +x ~/Downloads/NVIDIA-Linux-x86_64-367.57.run
	# from http://louwrentius.com/unattended-automatic-installation-of-linux-nvidia-binary-driver.html
	sudo sh ~/Downloads/NVIDIA-Linux-x86_64-367.57.run -q -a -n -X -s --no-opengl-files
	# sudo reboot
	wget https://developer.nvidia.com/compute/cuda/8.0/prod/local_installers/cuda_8.0.44_linux-run
	chmod +x cuda_8.0.44_linux-run
	sudo sh cuda_8.0.44_linux-run --extract=~/Downloads/
	sudo sh cuda_8.0.44_linux-run -silent -override -no-opengl-libs
	bash ./LD-path.sh
	# Extract and install CuDNN. this creates a `cuda` folder.
	sudo tar xopf cudnn-8.0-linux-x64-v6.0.tar
	sudo mkdir -p /usr/local/cuda
	sudo cp cuda/include/cudnn.h /usr/local/cuda/include
	sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64
	sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*
	# set env variables

build-tensorflow:
	bash ./tensorflow-compile.sh
install-tensorflow-wheel:
	bash ./tensorflow-install-wheel.sh

get-mujoco-id:
	wget https://www.roboti.us/getid/getid_linux
	chmod +x ./getid_linux
	./getid_linux
	rm ./getid_linux

get-mujoco-id-osx:
	wget https://www.roboti.us/getid/getid_osx
	chmod +x ./getid_osx
	./getid_osx
	rm ./getid_osx

get-mujoco:
	wget -O mjpro$(MJC_VER).zip https://www.roboti.us/download/mjpro$(MJC_VER)_linux.zip
	sudo apt-get install -y unzip
	mkdir -p ~/.mujoco
	rm -rf ~/.mujoco/mjpro$(MJC_VER)
	unzip mjpro$(MJC_VER).zip -d ~/.mujoco
	rm -rf mjpro$(MJC_VER).zip

set-mujoco-license:
	echo "now put your mjkey.txt file into ~/.mujoco/mjkey.txt"
	sudo vim ~/.mujoco/mjkey.txt

install-mujoco-dependencies:
	sudo apt-get update -q
	DEBIAN_FRONTEND=noninteractive sudo apt-get install -y \
    curl \
    git \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglapi-mesa \
    libosmesa6 \
    libosmesa6-dev \
    python3-pip \
    python3-numpy \
    python3-scipy \
    unzip \
    vim \
    wget \
    xpra \
    xserver-xorg-dev
	sudo apt-get clean
	sudo rm -rf /var/lib/apt/lists/*
# install patch
	sudo curl -o /usr/local/bin/patchelf https://s3-us-west-2.amazonaws.com/openai-sci-artifacts/manual-builds/patchelf_0.9_amd64.elf
	sudo chmod +x /usr/local/bin/patchelf
install-glfw:
	sudo apt-get update
	sudo apt-get install cmake xorg-dev libglu1-mesa-dev -y
	bash -c "source activate gym && pip install glfw --ignore-installed"
	rm -rf glfw
	git clone https://github.com/glfw/glfw.git
	bash -c "cd glfw && cmake -DBUILD_SHARED_LIBS=ON . && sudo make install"

install-mujoco-py:
	# Install Dependencies
	make install-glfw
	sudo apt-get update
	sudo apt-get install libjpeg-dev -y
	bash -c "source activate gym && pip install pillow"
	# installing with pip simply doesn't work.
	# sudo pip3 install -U 'mujoco-py==1.50.1.0'
	rm -rf mujoco-py
	git clone https://github.com/openai/mujoco-py.git
	bash -c "source activate gym && cd mujoco-py && python setup.py install"


install-gym:
	# Install Dependencies
#	sudo apt-get install -y python-numpy python-dev cmake zlib1g-dev libjpeg-dev xvfb libav-tools xorg-dev python-opengl libboost-all-dev libsdl2-dev swig
	# Install gym
#	rm -rf gym
#	git clone https://github.com/openai/gym.git
	# Do not use sudo with bash, then you can't use source activate.
#	bash -c "source activate gym && cd gym && pip install -e . && pip list --format=legacy | grep gym"
	# Do not use pip install -e, it relies on this local version
	# slight race condition for the pip list. First time install does not show updated location for gym
	bash -c "source activate gym && cd gym && python setup.py install && pip install '.[all]' && pip list --format=legacy | grep gym"
#	rm -rf gym


install-pytorch:
	bash -c "source activate gym && conda install pytorch torchvision -c soumith"


