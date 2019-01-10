#!/usr/bin/env bash

# It will tell you what drivers are needed
sudo ubuntu-drivers devices

# Install the drivers
sudo apt install '<the driver name>'

# Check the installations
nvidia-smi

# then try nvidia-smi to see if the drivers are installed properly if doesn’t work try using sudo with it and if it
# works then it means there are permission issues and it can be resolved with the following command
sudo apt install nvidia-modprobe

# Install cuda toolkit
sudo apt install nvidia-cuda-toolkit
#(For specific versions go to the Nvidia website and then download the coda and it also has the installation directions)

# Find out about your GPU make
cat /proc/driver/nvidia/gpus/0000\:01\:00.0/information

# find the hard disk
 sudo blkid | grep ok

# then mount it
udisksctl mount -b "the name you will find it from the above command"
# eg: ‘udisksctl mount -b /dev/sda2’

# unmount the disk
unmont "the name you will find it from the above command"
# eg: unmount /dev/sda2

Basic configuration
Cuda : 10.0
CuDNN: 7.4.2
NCCl : 2.3.7
Tensor flow: 1.12.0
Ubuntu : 18.04