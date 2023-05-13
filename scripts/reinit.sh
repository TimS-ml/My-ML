#!/bin/bash

# conda
/notebooks/miniconda3/bin/conda init  && source ~/.bashrc

# dot files
wget https://go.momar.de/bashrc -O ~/.bashrc
wget https://raw.githubusercontent.com/TimS-ml/My-ML/master/scripts/dots/.vimrc -O ~/.vimrc

# fzf
git clone --depth 1 https://github.com/junegunn/fzf.git ~/.fzf
~/.fzf/install

