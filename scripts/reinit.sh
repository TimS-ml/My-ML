#!/bin/bash

# dot files
# wget https://raw.githubusercontent.com/TimS-ml/My-ML/master/scripts/dots/.bashrc -O ~/.bashrc
wget https://raw.githubusercontent.com/TimS-ml/My-ML/master/scripts/dots/.vimrc -O ~/.vimrc

# fzf
# git clone --depth 1 https://github.com/junegunn/fzf.git ~/.fzf
# ~/.fzf/install

eval "$(conda shell.bash hook)"
conda activate /notebooks/env
export PYTHONPATH="${PYTHONPATH}:/notebooks/env/"
export PYTHONPATH="${PYTHONPATH}:/notebooks/env/utils/"

