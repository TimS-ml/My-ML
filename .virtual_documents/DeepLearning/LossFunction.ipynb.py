import torch
import torch.nn as nn
import torch.nn.functional as F


tensor_in = torch.randn(3, 5, requires_grad=True)
tensor_out = torch.randn(3, 5)


mae_loss = nn.L1Loss()
output = mae_loss(tensor_in, tensor_out)
output.backward()

print('tensor_in: ', tensor_in)
print('tensor_out: ', tensor_out)
print('output: ', output)


mse_loss = nn.MSELoss()
output = mse_loss(tensor_in, tensor_out)
output.backward()

print('tensor_in: ', tensor_in)
print('tensor_out: ', tensor_out)
print('output: ', output)


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


tensor_out_2 = torch.empty(3, dtype=torch.long).random_(5)

cross_entropy_loss = nn.CrossEntropyLoss()
output = cross_entropy_loss(tensor_in, tensor_out_2)
output.backward()

print('tensor_in: ', tensor_in)
print('tensor_out: ', tensor_out_2)
print('output: ', output)


hinge_loss = nn.HingeEmbeddingLoss()
output = hinge_loss(tensor_in, tensor_out)
output.backward()

print('tensor_in: ', tensor_in)
print('tensor_out: ', tensor_out)
print('output: ', output)


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


tensor_in = torch.randn(2, 3, requires_grad=True)
tensor_out_4 = torch.randn(2, 3)

kl_loss = nn.KLDivLoss(reduction = 'batchmean')
output = kl_loss(tensor_in, tensor_out_4)
output.backward()

print('tensor_in: ', tensor_in)
print('tensor_out: ', tensor_out_4)
print('output: ', output)


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
