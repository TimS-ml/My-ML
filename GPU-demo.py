# https://pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html#sphx-glr-beginner-blitz-tensor-tutorial-py

from __future__ import print_function
import torch

x = torch.randn(1)
# print(x)
# print(x.item())

# let us run this cell only if CUDA is available
# We will use ``torch.device`` objects to move tensors in and out of GPU
if torch.cuda.is_available():
    device = torch.device("cuda")          # a CUDA device object
    y = torch.ones_like(x, device=device)  # directly create a tensor on GPU
    x = x.to(device)                       # or just use strings ``.to("cuda")``
    z = x + y
    print('Has GPU: ', z)
    print('To CPU : ', z.to("cpu", torch.double))       # ``.to`` can also change dtype together!
