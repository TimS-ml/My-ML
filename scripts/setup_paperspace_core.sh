
# https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda
sudo apt-get -y install nvidia-utils-525 nvidia-cuda-toolkit

echo 'export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}' >> ~/.bashrc


# https://computingforgeeks.com/how-to-install-docker-on-ubuntu/
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/trusted.gpg.d/docker-archive-keyring.gpg
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"

# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#installation-guide
sudo apt-get update \
    && sudo apt-get install -y nvidia-container-toolkit-base


# docker build -t drug .
# docker image tag drug timsml/drug:latest
# docker image push timsml/drug:latest

