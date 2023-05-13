" ===== Basic =====
set nocompatible

" set showtabline =0
set termguicolors          " enable true colors support
set t_Co=256
set mouse=a
set autoread               " set auto reload

" filetype plugin on         " vimwiki
" filetype plugin indent on  " no need for nvim Load plugins according to detected filetype.
syntax on                  " Enable syntax highlighting.

set autoindent             " Indent according to previous line.
set expandtab              " Use spaces instead of tabs.
set softtabstop =4         " Tab key indents by 4 spaces.
set shiftwidth  =4         " >> indents by 4 spaces.
set shiftround             " >> indents to next multiple of 'shiftwidth'.
set ts          =4
set number
set relativenumber

set spelllang   =en_us
set backspace   =indent,eol,start  " Make backspace work as you would expect.
set hidden                 " Switch between buffers without having to save first.
set laststatus  =2         " Always show statusline.
set display     =lastline  " Show as much as possible of the last line.

set showmode               " Show current mode in command-line.
set showcmd                " Show already typed keys when more are expected.
set showmatch

set incsearch              " Highlight while searching with / or ?.
set hlsearch               " Keep matches highlighted.

set ttyfast                " Faster redrawing.
set lazyredraw             " Only redraw when necessary.

set splitbelow             " Open new windows below the current window.
set splitright             " Open new windows right of the current window.

set cursorline             " Find the current line quickly.
hi cursorline   ctermbg=darkred guibg=darkred
set wrapscan               " Searches wrap around end-of-file.
set report      =0         " Always report changed lines.
set synmaxcol   =200       " Only highlight the first 200 columns.

set list                   " Show non-printable characters.

set wildmenu
set wildmode    =list:longest
set wildignore  +=*/tmp/*,*.so,*.swp,*.zip
set path        +=**
set formatoptions-=cro
" au FileType * setlocal fo-=c fo-=r fo-=o

if has('multi_byte') && &encoding ==# 'utf-8'
  let &listchars = 'tab:▸ ,extends:❯,precedes:❮,nbsp:±'
else
  let &listchars = 'tab:> ,extends:>,precedes:<,nbsp:.'
endif

" Avoid fish shell breaks things that use 'shell'.
if &shell =~# 'fish$'
  set shell=/bin/zsh
endif

" Fold
set foldmethod=indent
set foldnestmax=10
set nofoldenable
set foldlevel=2

" Leader Key
nnoremap <SPACE> <Nop>
let mapleader=" "
let maplocalleader = ','

" Clipboard
set clipboard=unnamed
" nmap d "_d
" xmap <leader>p "_dP
xmap <localleader>p "_dP

nmap <Leader>f :%s///g<Left><Left><Left>
nmap <Leader>F :%s/\<\>//g<Left><Left><Left><Left><Left>
" nmap <C-t> :%s///g<Left><Left><Left>

" statusline -----
set statusline=
set statusline+=\ %f\ 
set statusline+=\ %m
" set statusline+=%#CursorColumn#
" set statusline+=%#LineNr#
set statusline+=%#PmenuSel#
set statusline+=%=
set statusline+=%#CursorColumn#
set statusline+=\ %y
" set statusline+=\ %{&fileencoding?&fileencoding:&encoding}
" set statusline+=\[%{&fileformat}\]
set statusline+=\ %p%%
set statusline+=\ %l:%c
set statusline+=\ 

" ===== Language =====
" cpp & java & r & md
autocmd BufRead,BufNewFile *.cpp,*.h,*java,*.r,*.R,*.rmd,*.Rmd,*.md setlocal tabstop=2 shiftwidth=2 softtabstop=2
" autocmd BufRead,BufNewFile *.snippets setlocal tabstop=4 shiftwidth=4 softtabstop=4

" R
" au BufRead,BufNewFile *.Rmd  setf rmd
" au BufRead,BufNewFile *.R  setf r

autocmd FileType python map <buffer> <F9> :w<CR>:exec '!python3' shellescape(@%, 1)<CR>
autocmd FileType java map <buffer> <F9> :w<CR>:exec '!javac' shellescape(@%, 1)<CR>
autocmd FileType cpp map <buffer> <F9> :w<CR>:exec '!g++ -O -Wall' shellescape(@%, 1)<CR>
