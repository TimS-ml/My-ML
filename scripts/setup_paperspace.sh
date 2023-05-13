#!/bin/bash

# vscode
curl -Lk 'https://code.visualstudio.com/sha/download?build=stable&os=cli-alpine-x64' --output vscode_cli.tar.gz && tar -xf vscode_cli.tar.gz

# conda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# dot files
wget https://go.momar.de/bashrc -O ~/.bashrc                                                    #
wget https://github.com/TimS-ml/My-ML/scripts/dots/.vimrc ~/.vimrc

# fzf
git clone --depth 1 https://github.com/junegunn/fzf.git ~/.fzf
~/.fzf/install

# apt
# apt install ranger nvim tmux
