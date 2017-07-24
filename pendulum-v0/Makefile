CURRENT_BRANCH=`git branch | grep \* | cut -d ' ' -f2`
RUN_BRANCH=escher-run-$${RUN_ID}
RUN_REMOTE=origin
#RUN_REMOTE_URL=`git remote $(CURRENT_BRANCH) get-url | grep \* | cut -d ' ' -f2`
#RUN_REMOTE_URL=https://github.com/escher-ai/sample-runs.git

default:
#	make clean
	make run
clean:
	rm -rf /tmp/tensorflow/pendulum-v0
start-tensorboard:
	tensorboard --logdir=/tmp/tensorflow
run:
	bash -c "source activate gym && \
	RUN_ID=$$(date +'%Y%m%d-%H%M%S') \
	DISPLAY=:1.0 python policy_gradient.py"
test:
	echo $(CURRENT_BRANCH)
	echo $(RUN_BRANCH)
	sleep 2
	echo $(RUN_BRANCH)
setup-escher-remote:
	CURRENT_BRANCH=$(CURRENT_BRANCH) && \
	RUN_REMOTE_URL=$(RUN_REMOTE_URL) && \
	bash -c "git remote add escher $${RUN_REMOTE_URL} &>/dev/null
	git push escher master
pre-run:
	CURRENT_BRANCH=$(CURRENT_BRANCH) && \
	RUN_ID=$$(date +'%Y%m%d-%H%M%S') \
	RUN_BRANCH=$(RUN_BRANCH) && \
	git checkout -b $${RUN_BRANCH} && \
	git add . && \
	git commit -m $$RUN_BRANCH} && \
	git push $(RUN_REMOTE) $(RUN_BRANCH) -f && \
	git checkout $${CURRENT_BRANCH} && \
	git merge --no-commit --squash ${RUN_BRANCH} && \
	git reset HEAD # to unstage the changes && \
	git branch -D $${RUN_BRANCH}
