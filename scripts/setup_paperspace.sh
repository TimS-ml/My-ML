#!/bin/bash

# repo
TOKEN=''
git clone https://${TOKEN}@github.com/TimS-ml/FREED

# vscode
curl -Lk 'https://code.visualstudio.com/sha/download?build=stable&os=cli-alpine-x64' --output vscode_cli.tar.gz && tar -xf vscode_cli.tar.gz

# conda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash ./Miniconda3-latest-Linux-x86_64.sh -b -p /notebooks/miniconda3/
source /notebooks/miniconda3/bin/activate

# dot files
wget https://raw.githubusercontent.com/TimS-ml/My-ML/master/scripts/dots/enviroment.yml -O .
wget https://raw.githubusercontent.com/TimS-ml/My-ML/master/scripts/dots/.bashrc -O ~/.bashrc
wget https://raw.githubusercontent.com/TimS-ml/My-ML/master/scripts/dots/.vimrc -O ~/.vimrc

# fzf
# git clone --depth 1 https://github.com/junegunn/fzf.git /notebooks/.fzf
# /notebooks/.fzf/install

# apt
# apt install ranger nvim tmux

/notebooks/miniconda3/bin/conda init  && source ~/.bashrc

cd /notebooks/FREED/scripts
conda env create -f environment_freed.yml

cp ~/.bashrc /notebooks/bashrc
