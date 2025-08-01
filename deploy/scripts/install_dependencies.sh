#!/bin/bash

# Ensure the script runs in non-interactive mode
export DEBIAN_FRONTEND=noninteractive

# Update the package list
sudo apt-get update

# install docker
sudo apt-get install -y docker.io

# start the docker and enable it 
sudo systemctl start docker 
sudo systemctl enable docker 

# install neccessary utilities
sudo apt-get install -y unzip curl 

# download and install the aws cli
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "/home/ubuntu/awscliv2.zip"
unzip -o /home/ubuntu/awscliv2.zip -d /home/ubuntu/
sudo /home/ubuntu/aws/install 

# add 'ubuntu' user to the docker group to run docker commands without sudo
sudo usermod -aG docker ubuntu

# clean up the aws cli installation files
rm -rf /home/ubuntu/awscliv2.zip /home/ubuntu/aws

