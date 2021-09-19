# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown] id="9QXsrr6Mp5e_"
# # MNIST Model

# %% id="NJD5dgd7zEMg"
from __future__ import print_function
# import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


# %% id="YBGOnz5NpiTw"
class Net(nn.Module):
    def __init__(self, mnist=True):
        super(Net, self).__init__()
        if mnist:
            num_channels = 1
        else:
            num_channels = 3

        self.conv1 = nn.Conv2d(num_channels, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


# %% [markdown] id="1EWDw3bip8Ie"
# # Training


# %% id="ujzd_d1kp_sX" colab={"base_uri": "https://localhost:8080/", "height": 1000} outputId="5cc8d333-95a2-4757-b6c1-9b040e60664c"
def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % args["log_interval"] == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(
                output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(
                dim=1,
                keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))


def main():
    batch_size = 64
    test_batch_size = 64
    epochs = 10
    lr = 0.01
    momentum = 0.5
    seed = 1
    log_interval = 500
    save_model = False
    no_cuda = False

    use_cuda = not no_cuda and torch.cuda.is_available()

    torch.manual_seed(seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(datasets.MNIST(
        '../data',
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307, ), (0.3081, ))
        ])),
        batch_size=batch_size,
        shuffle=True,
        **kwargs)

    test_loader = torch.utils.data.DataLoader(datasets.MNIST(
        '../data',
        train=False,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307, ), (0.3081, ))
        ])),
        batch_size=test_batch_size,
        shuffle=True,
        **kwargs)

    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    args = {}
    args["log_interval"] = log_interval
    for epoch in range(1, epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)

    if (save_model):
        torch.save(model.state_dict(), "mnist_cnn.pt")

    return model


model = main()

# %% [markdown] id="vDkkrT2prCU9"
# # Quantisation of Network

# %% [markdown] id="vFM8UV9CreIX"
# ## Quantisation Functions

# %% id="iCsoFvwLrgdu"
from collections import namedtuple
import torch
import torch.nn as nn

QTensor = namedtuple('QTensor', ['tensor', 'scale', 'zero_point'])


def calcScaleZeroPoint(min_val, max_val, num_bits=8):
    # Calc Scale and zero point of next
    qmin = 0.
    qmax = 2.**num_bits - 1.

    scale = (max_val - min_val) / (qmax - qmin)

    initial_zero_point = qmin - min_val / scale

    zero_point = 0
    if initial_zero_point < qmin:
        zero_point = qmin
    elif initial_zero_point > qmax:
        zero_point = qmax
    else:
        zero_point = initial_zero_point

    zero_point = int(zero_point)

    return scale, zero_point


def quantize_tensor(x, num_bits=8, min_val=None, max_val=None):
    if not min_val and not max_val:
        min_val, max_val = x.min(), x.max()

    qmin = 0.
    qmax = 2.**num_bits - 1.

    scale, zero_point = calcScaleZeroPoint(min_val, max_val, num_bits)
    q_x = zero_point + x / scale
    q_x.clamp_(qmin, qmax).round_()
    q_x = q_x.round().byte()

    return QTensor(tensor=q_x, scale=scale, zero_point=zero_point)


def dequantize_tensor(q_x):
    return q_x.scale * (q_x.tensor.float() - q_x.zero_point)


# %% [markdown] id="xXkTAJ9ws1Y6"
# ## Rework Forward pass of Linear and Conv Layers to support Quantisation


# %% id="M5xNLrchrI6u"
def quantizeLayer(x, layer, stat, scale_x, zp_x):
    # for both conv and linear layers

    # cache old values
    W = layer.weight.data
    B = layer.bias.data

    # quantise weights, activations are already quantised
    w = quantize_tensor(layer.weight.data)
    b = quantize_tensor(layer.bias.data)

    layer.weight.data = w.tensor.float()
    layer.bias.data = b.tensor.float()

    # This is Quantisation Artihmetic
    scale_w = w.scale
    zp_w = w.zero_point
    scale_b = b.scale
    zp_b = b.zero_point

    scale_next, zero_point_next = calcScaleZeroPoint(min_val=stat['min'],
                                                     max_val=stat['max'])

    # Preparing input by shifting
    X = x.float() - zp_x
    layer.weight.data = scale_x * scale_w * (layer.weight.data - zp_w)
    layer.bias.data = scale_b * (layer.bias.data + zp_b)

    # All int computation
    x = (layer(X) / scale_next) + zero_point_next

    # Perform relu too
    x = F.relu(x)

    # Reset weights for next forward pass
    layer.weight.data = W
    layer.bias.data = B

    return x, scale_next, zero_point_next


# %% [markdown] id="OgkWg605tE1y"
# ## Get Max and Min Stats for Quantising Activations of Network.
#
# This is done by running the network with around 1000 examples and getting the average min and max activation values before and after each layer.


# %% id="GecOkNLhtVh9"
# Get Min and max of x tensor, and stores it
def updateStats(x, stats, key):
    max_val, _ = torch.max(x, dim=1)
    min_val, _ = torch.min(x, dim=1)

    if key not in stats:
        stats[key] = {"max": max_val.sum(), "min": min_val.sum(), "total": 1}
    else:
        stats[key]['max'] += max_val.sum().item()
        stats[key]['min'] += min_val.sum().item()
        stats[key]['total'] += 1

    return stats


# Reworked Forward Pass to access activation Stats through updateStats function
def gatherActivationStats(model, x, stats):
    stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'conv1')

    x = F.relu(model.conv1(x))
    x = F.max_pool2d(x, 2, 2)
    stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'conv2')

    x = F.relu(model.conv2(x))
    x = F.max_pool2d(x, 2, 2)
    x = x.view(-1, 4 * 4 * 50)
    stats = updateStats(x, stats, 'fc1')

    x = F.relu(model.fc1(x))
    stats = updateStats(x, stats, 'fc2')

    x = model.fc2(x)
    return stats


# Entry function to get stats of all functions.
def gatherStats(model, test_loader):
    device = 'cuda'

    model.eval()
    # test_loss = 0
    # correct = 0
    stats = {}
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            stats = gatherActivationStats(model, data, stats)

    final_stats = {}
    for key, value in stats.items():
        final_stats[key] = {
            "max": value["max"] / value["total"],
            "min": value["min"] / value["total"]
        }
    return final_stats


# %% [markdown] id="OBt0WDzyujnk"
# ## Forward Pass for Quantised Inference


# %% id="f6duGNF_uoZB"
def quantForward(model, x, stats):

    # Quantise before inputting into incoming layers
    x = quantize_tensor(x,
                        min_val=stats['conv1']['min'],
                        max_val=stats['conv1']['max'])

    x, scale_next, zero_point_next = quantizeLayer(x.tensor, model.conv1,
                                                   stats['conv2'], x.scale,
                                                   x.zero_point)

    x = F.max_pool2d(x, 2, 2)

    x, scale_next, zero_point_next = quantizeLayer(x, model.conv2,
                                                   stats['fc1'], scale_next,
                                                   zero_point_next)

    x = F.max_pool2d(x, 2, 2)

    x = x.view(-1, 4 * 4 * 50)

    x, scale_next, zero_point_next = quantizeLayer(x, model.fc1, stats['fc2'],
                                                   scale_next, zero_point_next)

    # Back to dequant for final layer
    x = dequantize_tensor(
        QTensor(tensor=x, scale=scale_next, zero_point=zero_point_next))

    x = model.fc2(x)

    return F.log_softmax(x, dim=1)


# %% [markdown] id="xC96eesMqYo-"
# # Testing Function for Quantisation


# %% id="X6jKRKSBt0he"
def testQuant(model, test_loader, quant=False, stats=None):
    device = 'cuda'

    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            if quant:
                output = quantForward(model, data, stats)
            else:
                output = model(data)
            test_loss += F.nll_loss(
                output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(
                dim=1,
                keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))


# %% [markdown] id="bs97rNEXt_my"
# # Get Accuracy of Non Quantised Model

# %% id="0YCtbfk9qbGI"
import copy
q_model = copy.deepcopy(model)

# %% id="5j42Q8PKt3lj"
kwargs = {'num_workers': 1, 'pin_memory': True}
test_loader = torch.utils.data.DataLoader(datasets.MNIST(
    '../data',
    train=False,
    transform=transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.1307, ), (0.3081, ))])),
                                          batch_size=64,
                                          shuffle=True,
                                          **kwargs)

# %% id="QeYlzGG0t4Yp" colab={"base_uri": "https://localhost:8080/", "height": 70} outputId="e62bcadf-5c3c-416a-c125-501abc6ef9b7"
testQuant(q_model, test_loader, quant=False)

# %% [markdown] id="1JaeISHeuHCb"
# # Gather Stats of Activations

# %% id="xhiL7OwwuLS6" colab={"base_uri": "https://localhost:8080/", "height": 54} outputId="8df4a58c-64e7-4b8e-e98c-349c0212e210"
stats = gatherStats(q_model, test_loader)
print(stats)

# %% [markdown] id="eMeng9S4uSOX"
# # Test Quantised Inference Of Model

# %% id="INQggUUQuXyq" colab={"base_uri": "https://localhost:8080/", "height": 70} outputId="bbf830f0-60c5-4519-825e-0a996dc86b75"
testQuant(q_model, test_loader, quant=True, stats=stats)

# %% [markdown] id="voLb1LPkvkz_"
# ## TA DA !!
#
# We have quantised our net to mostly 8 bit arithmetic with almost zero accuracy loss ! Pretty good day's work, I'll say :D
