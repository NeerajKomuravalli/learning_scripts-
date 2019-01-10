#!/usr/bin/env bash

# Find a file and with a particular pattern an d then send it to destination
find ./ -name '*.out' | xargs -I '{}' mv {} destionantion

# https://stackoverflow.com/questions/19071512/socket-error-errno-48-address-already-in-use
ps -fA | grep python

# To see what programmes are using your GPU
sudo fuser -v /dev/nvidia*

# Submodule commands
# Add a submodule
git submodule add submodule_git_link

# After you add a submodule commit and push it
git commit -am 'added new sub module'
git push origin master

# To make sure, when we pull we pull both the main and sub module repo
git clone --recurse-submodules main_project_git_link

# If you want to update the submodule
#cd <submodule>
git fetch
git merge origin/master
#(Or)
git submodule update --remote submodule_name
git add
Git commit -m 'Updating the submodule'
# Remove a submodule
git submodule deinit path_to_submodule
git rm path_to_submodule
git commit-m "Removed submodule "
rm -rf .git/modules/path_to_submodule
# In case of nested submodules use this
git submodule update --init --recursive

# AWS (using s3cmd)
# configure
s3cmd --configure
# to make a folder in s3 bucket private
s3cmd setacl --acl-private --recursive s3://mybucket.com/topleveldir/scripts/
# To make the same public
s3cmd setacl --acl-public --recursive s3://mybucket.com/topleveldir/scripts/

# Lambda AWS
aws lambda create-function --function-name identity_extraction_cli --code S3Bucket=identity-bucket,S3Key=text_extract.zip --runtime python3.6 --role arn:aws:iam::609891413446:role/identity_extraction_lambda_role --handler lambda_testing.lambda_handler
aws lambda invoke --invocation-type RequestResponse --function-name arn:aws:lambda:us-east-1:609891413446:function:identity_extraction_cli --payload '{"key1":"value1", "key2":"value2", "key3":"value3"}' output.txt


# virtual env on python3
python3 -m venv '<env_name>'

# Docker
docker build -t friendlyhello .
# List all the images
docker image ls
docker rmi -f '<IMAGEID>'

#To show only running containers use the given command:
docker ps
#To show all containers use the given command:
docker ps -a
#To show the latest created container (includes all states) use the given command:
docker ps -l
#To show n last created containers (includes all states) use the given command:
docker ps -n=-1
#To display total file sizes use the given command:
docker ps -s
#In the new version of Docker, commands are updated, and some management commands are added:
docker container ls
#Is used to list all the running containers.
docker container ls -a
# To run docker
docker run -d '<IMAGE>'
#To stop
docker stop '<containerID>'
# Save
docker save -o '<path for generated tar file>' '<image name>'
#Load
docker load -i '<path to image tar file>'
# Use docker without sudo (happens in linux systems)
sudo setfacl -m user:username:rw /var/run/docker.sock

# # GPU using docker
# for python2-gpu version but as we are using -it it will be a interactive and won't save anything when you exit it
docker run --runtime=nvidia -it tensorflow/tensorflow:latest-gpu bash
# The below commands will build and run it and then later you can enter by using exec command and start and stop by
# docker start and docker stop command
# Below are to just buid it
# For python2-gpu version
sudo docker run --runtime=nvidia -td tensorflow/tensorflow:latest-gpu bash
# For python3-gpu version
sudo docker run --runtime=nvidia -td tensorflow/tensorflow:latest-gpu-py3 bash
# For python3-gpu version with media and give a name
sudo docker run -v /media:/media --name tensorflow_py3_gpu_media --runtime=nvidia -td tensorflow/tensorflow:latest-gpu-py3
# For executing (or entering the bash of the docker)
sudo docker exec -it [container-id] bash


# Change git from ssh to https or other way round
git remote set-url origin '<ssh link or https link>'