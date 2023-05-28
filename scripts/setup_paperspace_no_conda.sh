#!/bin/bash

# repo
TOKEN=''
git clone https://${TOKEN}@github.com/apostolosfilippas/deep

# vscode
curl -Lk 'https://code.visualstudio.com/sha/download?build=stable&os=cli-alpine-x64' --output vscode_cli.tar.gz && tar -xf vscode_cli.tar.gz

# dot files
wget https://raw.githubusercontent.com/TimS-ml/My-ML/master/scripts/dots/.bashrc -O ~/.bashrc
wget https://raw.githubusercontent.com/TimS-ml/My-ML/master/scripts/dots/.vimrc -O ~/.vimrc

# fzf
git clone --depth 1 https://github.com/junegunn/fzf.git /notebooks/.fzf
/notebooks/.fzf/install

# apt
# apt install ranger nvim tmux

cp ~/.bashrc /notebooks/bashrc
