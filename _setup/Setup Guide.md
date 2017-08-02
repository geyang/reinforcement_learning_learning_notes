# Homework Environment Setup Guide

All of the scripts needed are inside the `Makefile`. This tutorial is designed for OSX and linux. 

## First install MuJoCo

**Requirements**: you need to have the license file ready. This script is going to ask you for your 
license file at the end. To obtain the hardware key for student license, first run `make install-mujoco` 
inside the `_setup` folder.

Now with the license file in your hand, run 
```
cd _setup
make install-mujoco
```
The script uses vim to open up the license file in the end. If you don't know how to get out of the vim file, take a look at this link: [how to](https://stackoverflow.com/questions/11828270/how-to-exit-the-vim-editor).

## Compile and install Tensorflow

This will take an hour, you can go ahead and jump to next while waiting.

#