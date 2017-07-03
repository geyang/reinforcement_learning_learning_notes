include ./.config

install:
	pip install -r requirements.txt
init:
	source ./env
start-display:
	Xvfb :1 -screen 0 1400x900x24 +extension RANDR &
create-vnc-password:
	x11vnc -storepasswd $(VNC_PASSWORD) /tmp/vncpass
start-vnc:
	x11vnc -ncache -rfbport $(VNC_PORT) -rfbauth /tmp/vncpass -display :1 -forever -auth /tmp/xvfb.auth
# now on local computer
build-tunnel:
	ssh -L $(VNC_PORT):localhost:$(VNC_PORT) ubuntu@$(IP_ADDRESS) -i ~/.ec2/escherpad.pem
	x11vnc -safer -localhost -nopw -once -display :1.0
run-script:
	xvfb-run -a -s "-screen 0 1400x900x24 +extension RANDR" -- python3 cartpole.py

# Installation Scripts
install-conda:
	wget https://repo.continuum.io/archive/Anaconda3-4.4.0-Linux-x86_64.sh
	sudo bash ./Anaconda3-4.4.0-Linux-x86_64.sh
install-python3.5.2:
	conda create --name gym python=3.5.2
install-tensorflow-wheel:
	pip install /tmp/tensorflow_pkg/tensorflow-1.2.0-cp35-cp35m-linux_x86_64.whl
get-mujoco-id:
	wget https://www.roboti.us/getid/getid_linux
	chmod +x ./getid_linux
	./getid_linux
	rm ./getid_linux
get-mujoco:
	wget -O mjpro150.zip https://www.roboti.us/download/mjpro150_linux.zip
	sudo apt-get install unzip
	rm -rf ~/.mujoco
	mkdir ~/.mujoco
	unzip mjpro150.zip -d ~/.mujoco/mjpro150
	rm -rf mjpro150.zip
	echo "now put your `mjkey.txt` file into ~/.mujoco/mjkey.txt"
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

install-mujoco:
# installing with pip simply doesn't work.
#	sudo pip3 install -U 'mujoco-py==1.50.1.0'
	rm -rf mujoco-py
	git clone https://github.com/openai/mujoco-py.git
	bash -c "cd mujoco-py && source activate gym && python setup.py install"



