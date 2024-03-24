for pkg in $(pip list | grep 'llama-index' | awk '{print $1}'); do
    pip install --upgrade $pkg
done

