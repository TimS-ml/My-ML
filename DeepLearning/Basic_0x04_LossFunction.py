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
#       jupytext_version: 1.11.5
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
# ## L1Loss: Mean Absolute Error
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
# ## MSELoss: Mean Squared Error
# - MSE is the default loss function for most Pytorch regression problems.

# %%
mse_loss = nn.MSELoss()
output = mse_loss(tensor_in, tensor_out)
output.backward()

print('tensor_in: ', tensor_in)
print('tensor_out: ', tensor_out)
print('output: ', output)

# %% [markdown]
# ## NLLLoss: Negative Log-Likelihood Loss Function
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
# ## CrossEntropyLoss: Cross-Entropy
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
# ## HingeEmbeddingLoss: Hinge Embedding
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
# ## MarginRankingLoss: Margin Ranking
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
# ## TripletMarginLoss: Triplet Margin
# - Determining the relative similarity existing between samples. 
# - It is used in content-based retrieval problems 
#
# $$L(a, p, n) = \max \{d(a_i, p_i) - d(a_i, n_i) + {\rm margin}, 0\}$$
#
# $$d(x_i, y_i) = \left\lVert {\bf x}_i - {\bf y}_i \right\rVert_p$$

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
# ### Implement code
#
# https://pytorch.org/docs/stable/_modules/torch/nn/modules/loss.html#TripletMarginLoss
#
# https://discuss.pytorch.org/t/triplet-loss-in-pytorch/30634

# %%
class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin = 1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        distance = distance_positive - distance_negative + self.margin
        losses = F.relu(distance)  # only > 0
        return losses.mean() if size_average else losses.sum()


# %%
class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin = 1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        d = nn.PairwiseDistance(p=2)
        distance_positive = d(anchor, positive)
        distance_negative = d(anchor, negative)
        distance = distance_positive - distance_negative + self.margin
        losses = torch.max(distance, torch.zeros_like(distance))  # only > 0
        return losses.mean() if size_average else losses.sum()


# %%
class TripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.
    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.
    Args:
        margin (float): margin for triplet.
    """

    def __init__(self, margin=1.0, mutual_flag=False):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.mutual = mutual_flag

    def forward(self, inputs, targets):
        """
        Args:
            inputs: feature matrix with shape (batch_size, feat_dim)
            targets: ground truth labels with shape (num_classes)
        """
        n = inputs.size(0)
        # inputs = 1. * inputs / (torch.norm(inputs, 2, dim=-1, keepdim=True).expand_as(inputs) + 1e-12)
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        if self.mutual:
            return loss, dist
        return loss


# %% [markdown]
# ## KLDivLoss: Kullback-Leibler Divergence
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
    # specifying the batch size
    my_batch_size = my_outputs.size()[0] 

    my_outputs = F.log_softmax(my_outputs, dim=1)
    my_outputs = my_outputs[range(my_batch_size), my_labels]
    
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
