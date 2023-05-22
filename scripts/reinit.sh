#!/bin/bash

# dot files
# wget https://go.momar.de/bashrc -O ~/.bashrc
cp /notebooks/bashrc ~/.bashrc
source ~/.bashrc
# wget https://raw.githubusercontent.com/TimS-ml/My-ML/master/scripts/dots/.bashrc -O ~/.bashrc
wget https://raw.githubusercontent.com/TimS-ml/My-ML/master/scripts/dots/.vimrc -O ~/.vimrc

# conda
# /notebooks/miniconda3/bin/conda init  && source ~/.bashrc

# fzf
# git clone --depth 1 https://github.com/junegunn/fzf.git ~/.fzf
# ~/.fzf/install

