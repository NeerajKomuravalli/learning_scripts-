#!/usr/bin/env bash

# Find all the directories
ls -d */

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
# Remove a submodulecool
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
# In built with python3
python3 -m venv '<env_name>'
# virtualenv is a different package all together and would need installation
virtualenv '<env_name>'

# Docker
# Example
docker build -t friendlyhello .
# build from a folder containing the Dockerfile
docker image build kyc_docker/
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

# tagging your image
docker tag bb38976d03cf yourhubusername/verse_gapminder:firsttry


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

# SSH when runnign jupyter notebook
ssh -L 8000:localhost:8888 <ip address>
# example ssh -L 8000:localhost:8888 puzzle@puzzle.does-it.net

# AWS ECR 
# Referance link : https://docs.aws.amazon.com/AmazonECR/latest/userguide/docker-basics.html
# Create the repo (if not already present)
aws ecr create-repository --repository-name hello-repository --region region
# Tag the image you have already build on local machine
docker tag hello-world aws_account_id.dkr.ecr.region.amazonaws.com/hello-repository:tag_name
# Login to ecr (this will give you a command with token and it will be valid for 12 hours and use that command to actaully login succesfully)
aws ecr get-login --no-include-email --region region
# finally push the image
docker push aws_account_id.dkr.ecr.region.amazonaws.com/hello-repository:tag_name
# View all the images in a repo
aws ecr describe-images --repository-name foo 

# AWS ssm for secure keep of parameters
# storing using KMS
aws ssm put-parameter --name "Name-of-the-param" --value "value" --type "SecureString" --key-id alias/<name-of-the-KMS>
# Without KMS
aws ssm put-parameter --name "Name-of-the-param" --value "value" --type "String"
# With json 
aws ssm put-parameter --cli-input  '{"Name" : "Name-of-the-param", "Value" : "value"}' --type "SecureString" --key-id alias/verizy-crust

# To view all parameters
aws ssm describe-parameters --type "SecureString" --key-id alias/verizy-crust

# To get value of parameter and parameters respectively
aws ssm get-parameter --name "Name-of-the-param"
aws ssm get-parameters --names "Name-of-the-param1" "Name-of-the-param2"


# ffmpeg
ffmpeg -ss 00:01:00 -i input.mp4 -to 00:02:00 -c copy output.mp4
# About the command:
# -i: This specifies the input file. In that case, it is (input.mp4). 
# -ss: Used with -i, this seeks in the input file (input.mp4) to position. 
# 00:01:00: This is the time your trimmed video will start with. 
# -to: This specifies duration from start (00:01:40) to end (00:02:12). 
# 00:02:00: This is the time your trimmed video will end with. 
# -c copy: This is an option to trim via stream copy. (NB: Very fast) 

# Neo4j on mac
# To start neo4j server
~/neo4j-community-4.2.5/bin/neo4j start
# To stop neo4j server
~/neo4j-community-4.2.5/bin/neo4j stop
# To start cypher temrinal
~/neo4j-community-4.2.5/bin/cypher-shell -a bolt://localhost:7687 -u dbname -p password