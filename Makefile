IP=`< ip_address.txt`

ssh:
	ssh ubuntu@${IP} -i ~/.ec2/escherpad.pem
build-tunnel:
	ssh -L $(VNC_PORT):localhost:$(VNC_PORT) ubuntu@$(IP_ADDRESS) -i ~/.ec2/escherpad.pem
	x11vnc -safer -localhost -nopw -once -display :1.0
open-tensorboard:
	open http://${IP}:6006
clear:
	rm -rf /tmp/tensorflow/
#run-script:
#	xvfb-run -a -s "-screen 0 1400x900x24 +extension RANDR" -- python3 cartpole.py

# remote scripts
start-tensorboard:
	bash -c "source activate gym && tensorboard --logdir=/tmp/tensorflow/"
install-pdflatex:
	sudo apt-get install latexlive
start-jupyter:
	bash -c "source activate gym && DISPLAY=:1.0 jupyter lab --ip='*'"
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
