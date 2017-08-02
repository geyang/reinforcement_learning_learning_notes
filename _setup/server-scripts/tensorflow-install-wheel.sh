#!/usr/bin/env bash
# Only the compilation step for tensorflow is in this script, for clarity.

SEARCH_PATH=/tmp/tensorflow_pkg
# if multiple lines are found with the same tensorflow- prefix, they will
# be concatenated into a single space separated line. Potentially a corner 
# case to watch out for.
WHEEL_PATH=$SEARCH_PATH/$(ls -1 $SEARCH_PATH | grep tensorflow-)
echo $WHEEL_PATH
source activate gym && pip install $WHEEL_PATH
