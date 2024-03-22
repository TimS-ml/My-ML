curl -Lk 'https://code.visualstudio.com/sha/download?build=stable&os=cli-alpine-x64' --output vscode_cli.tar.gz && tar -xf vscode_cli.tar.gz
rm -f vscode_cli.tar.gz

# code --list-extensions
./code --install-extension bierner.markdown-mermaid
./code --install-extension edonet.vscode-command-runner
./code --install-extension github.copilot
./code --install-extension github.copilot-chat
./code --install-extension mechatroner.rainbow-csv
./code --install-extension modular-mojotools.vscode-mojo
./code --install-extension ms-azuretools.vscode-docker
./code --install-extension ms-python.debugpy
./code --install-extension ms-python.python
./code --install-extension ms-python.vscode-pylance
./code --install-extension ms-toolsai.jupyter
./code --install-extension ms-toolsai.jupyter-keymap
./code --install-extension ms-toolsai.jupyter-renderers
./code --install-extension ms-toolsai.vscode-jupyter-cell-tags
./code --install-extension ms-toolsai.vscode-jupyter-slideshow
./code --install-extension ms-vscode-remote.remote-containers
./code --install-extension ms-vscode-remote.remote-ssh
./code --install-extension ms-vscode-remote.remote-ssh-edit
./code --install-extension ms-vscode.cmake-tools
./code --install-extension ms-vscode.cpptools
./code --install-extension ms-vscode.cpptools-extension-pack
./code --install-extension ms-vscode.cpptools-themes
./code --install-extension ms-vscode.live-server
./code --install-extension ms-vscode.remote-explorer
./code --install-extension ms-vscode.remote-server
./code --install-extension tomoki1207.pdf
./code --install-extension twxs.cmake
./code --install-extension vscodevim.vim
./code --install-extension xelad0m.jupyter-toc

