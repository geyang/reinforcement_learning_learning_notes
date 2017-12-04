default:
	dave --config-file "./scripts/breakout.yml"
breakout:
	PYTHONPATH=/home/ubuntu/machine_learning/ge_readings_and_prep/ge_dqn python ./scripts/dqn_breakout.py
pull:
	aws s3 sync s3://ge_dqn /Users/ge/data/rcall/s3/ge_dqn
data:
	open /Users/ge/data/rcall/s3/ge_dqn
