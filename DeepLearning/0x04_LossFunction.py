# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     comment_magics: false
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python (torch)
#     language: python
#     name: torch
# ---

# %% [markdown]
# # Loss Function
# https://neptune.ai/blog/pytorch-loss-functions

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F


# %%
tensor_in = torch.randn(3, 5, requires_grad=True)
tensor_out = torch.randn(3, 5)

# %% [markdown]
# ## The Mean Absolute Error
# - Regression problems, especially when the distribution of the tensor_out variable has outliers, 
#   - such as small or big values that are a great distance from the mean value. It is considered to be more robust to outliers.

# %%
mae_loss = nn.L1Loss()
output = mae_loss(tensor_in, tensor_out)
output.backward()

print('tensor_in: ', tensor_in)
print('tensor_out: ', tensor_out)
print('output: ', output)

# %% [markdown]
# ## Mean Squared Error Loss Function
# - MSE is the default loss function for most Pytorch regression problems.

# %%
mse_loss = nn.MSELoss()
output = mse_loss(tensor_in, tensor_out)
output.backward()

print('tensor_in: ', tensor_in)
print('tensor_out: ', tensor_out)
print('output: ', output)

# %% [markdown]
# ## Negative Log-Likelihood Loss Function
# https://ljvmiranda921.github.io/notebook/2017/08/13/softmax-and-the-negative-log-likelihood/
#
# - Multi-class classification problems
#
# <img src="https://i.imgur.com/hU252jE.jpg" width="500">

# %%
# size of tensor_in (N x C) is = 3 x 5
# every element in tensor_out should have 0 <= value < C
# tensor_out_2 = torch.tensor([1, 0, 4])
tensor_out_2 = torch.tensor([2, 4, 4])

# (NLL) is applied only on models with the softmax function as an output activation layer. 
# Softmax refers to an activation function that calculates the normalized exponential function of every unit in the layer.
m = nn.LogSoftmax(dim=1)
nll_loss = nn.NLLLoss()
output = nll_loss(m(tensor_in), tensor_out_2)
output.backward()

print('tensor_in: ', tensor_in)
print('tensor_out: ', tensor_out_2)
print('output: ', output)

# %% [markdown]
# ## Cross-Entropy Loss Function
# - Common type is the Binary Cross-Entropy (BCE)
#   - The BCE Loss is mainly used for binary classification models
# - Creating confident models—the prediction will be accurate and with a higher probability

# %%
tensor_out_2 = torch.empty(3, dtype=torch.long).random_(5)

cross_entropy_loss = nn.CrossEntropyLoss()
output = cross_entropy_loss(tensor_in, tensor_out_2)
output.backward()

print('tensor_in: ', tensor_in)
print('tensor_out: ', tensor_out_2)
print('output: ', output)

# %% [markdown]
# ## Hinge Embedding Loss Function
# - Classification problems, especially when determining if two tensor_ins are dissimilar or similar. 
# - Learning nonlinear embeddings or semi-supervised learning tasks.

# %%
hinge_loss = nn.HingeEmbeddingLoss()
output = hinge_loss(tensor_in, tensor_out)
output.backward()

print('tensor_in: ', tensor_in)
print('tensor_out: ', tensor_out)
print('output: ', output)

# %% [markdown]
# ## Margin Ranking Loss Function
# - Ranking problems

# %%
import torch
import torch.nn as nn

tensor_in_one = torch.randn(3, requires_grad=True)
tensor_in_two = torch.randn(3, requires_grad=True)
tensor_out_3 = torch.randn(3).sign()

ranking_loss = nn.MarginRankingLoss()
output = ranking_loss(tensor_in_one, tensor_in_two, tensor_out_3)
output.backward()

print('tensor_in one: ', tensor_in_one)
print('tensor_in two: ', tensor_in_two)
print('tensor_out: ', tensor_out_3)
print('output: ', output)

# %% [markdown]
# ## Triplet Margin Loss Function
# - Determining the relative similarity existing between samples. 
# - It is used in content-based retrieval problems 

# %%
anchor = torch.randn(100, 128, requires_grad=True)
positive = torch.randn(100, 128, requires_grad=True)
negative = torch.randn(100, 128, requires_grad=True)

triplet_margin_loss = nn.TripletMarginLoss(margin=1.0, p=2)
output = triplet_margin_loss(anchor, positive, negative)
output.backward()

print('anchor: ', anchor)
print('positive: ', positive)
print('negative: ', negative)
print('output: ', output)

# %% [markdown]
# ## Kullback-Leibler Divergence Loss Function
# - Approximating complex functions
# - Multi-class classification tasks
# - If you want to make sure that the distribution of predictions is similar to that of training data

# %%
tensor_in = torch.randn(2, 3, requires_grad=True)
tensor_out_4 = torch.randn(2, 3)

kl_loss = nn.KLDivLoss(reduction = 'batchmean')
output = kl_loss(tensor_in, tensor_out_4)
output.backward()

print('tensor_in: ', tensor_in)
print('tensor_out: ', tensor_out_4)
print('output: ', output)


# %% [markdown]
# ## Custom Loss Function

# %%
# type 1
def myCustomLoss(my_outputs, my_labels):
    #specifying the batch size
    my_batch_size = my_outputs.size()[0] 
    #calculating the log of softmax values           
    my_outputs = F.log_softmax(my_outputs, dim=1)  
    #selecting the values that correspond to labels
    my_outputs = my_outputs[range(my_batch_size), my_labels] 
    #returning the results
    return -torch.sum(my_outputs)/number_examples

# type 2
# in binary classification problem
class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()
 
    def forward(self, tensor_ins, tensor_outs, smooth=1):        
        tensor_ins = F.sigmoid(tensor_ins)       
        
        tensor_ins = tensor_ins.view(-1)
        tensor_outs = tensor_outs.view(-1)
        
        intersection = (tensor_ins * tensor_outs).sum()                            
        dice = (2.*intersection + smooth)/(tensor_ins.sum() + tensor_outs.sum() + smooth)  
        
        return 1 - dice
