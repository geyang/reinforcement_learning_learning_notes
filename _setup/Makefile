include ./vnc.secretes.config

MJC_VER=131
MJC_DL_URL=https://www.roboti.us/download/mjpro131_linux.zip

gpu-monitor:
	watch -n1 "nvidia-smi"
install-x11vnc:
	sudo apt-get install -y x11vnc
create-vnc-password:
	x11vnc -storepasswd $(VNC_PASSWORD) /tmp/vncpass
# Only after running the above
start-display:
	Xvfb :1 -screen 0 1400x900x24 +extension RANDR &
start-vnc:
	sudo x11vnc -ncache -rfbport $(VNC_PORT) -rfbauth /tmp/vncpass -display :1 -forever -auth /tmp/xvfb.auth
start-tensorboard:
	tensorboard --logdir=/tmp/tensorflow/

# now on local computer
build-tunnel:
	ssh -L $(VNC_PORT):localhost:$(VNC_PORT) ubuntu@$(cat ip_address.txt) -i ~/.ec2/escherpad.pem
	x11vnc -safer -localhost -nopw -once -display :1.0
run-script:
	xvfb-run -a -s "-screen 0 1400x900x24 +extension RANDR" -- python3 cartpole.py

# Installation Scripts
install-conda:
	wget https://repo.continuum.io/archive/Anaconda3-4.4.0-Linux-x86_64.sh
	sudo bash ./Anaconda3-4.4.0-Linux-x86_64.sh

install-python3.5.2:
	conda create --name gym python=3.5.2

install:
	bash -c "source activate gym && pip install -r requirements.txt"

install-gym:
	rm -rf gym
	git clone https://github.com/openai/gym.git
	# Do not use sudo with bash, then you can't use source activate.
#	bash -c "source activate gym && cd gym && pip install -e . && pip list --format=legacy | grep gym"
	# Do not use pip install -e, it relies on this local version
	# slight race condition for the pip list. First time install does not show updated location for gym
	bash -c "source activate gym && cd gym && python setup.py install && pip install --install-option='--prefix=$PREFIX_PATH' '.[all]' && pip list --format=legacy | grep gym"
	rm -rf gym

install-tensorflow-wheel:
	pip install /tmp/tensorflow_pkg/tensorflow-1.2.0-cp35-cp35m-linux_x86_64.whl

get-mujoco-id:
	wget https://www.roboti.us/getid/getid_linux
	chmod +x ./getid_linux
	./getid_linux
	rm ./getid_linux

get-mujoco:
	wget -O mjpro$(MJC_VER).zip https://www.roboti.us/download/mjpro$(MJC_VER)_linux.zip
	sudo apt-get install unzip
	mkdir -p ~/.mujoco
	rm -rf ~/.mujoco/mjpro$(MJC_VER)
	unzip mjpro$(MJC_VER).zip -d ~/.mujoco
	rm -rf mjpro$(MJC_VER).zip
	echo "now put your mjkey.txt file into ~/.mujoco/mjkey.txt"
	sudo vim ~/.mujoco/mjkey.txt

install-mujoco-dependencies:
	sudo apt-get update -q
	DEBIAN_FRONTEND=noninteractive sudo apt-get install -y \
    curl \
    git \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
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
	bash -c "source activate gym && pip install glfw --ignore-installed"
	rm -rf glfw
	git clone https://github.com/glfw/glfw.git
	bash -c "cd glfw && cmake -DBUILD_SHARED_LIBS=ON . && sudo make install"
install-mujoco-py:
	# installing with pip simply doesn't work.
	# sudo pip3 install -U 'mujoco-py==1.50.1.0'
	rm -rf mujoco-py
	git clone https://github.com/openai/mujoco-py.git
	bash -c "source activate gym && cd mujoco-py && python setup.py install"



