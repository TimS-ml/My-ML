# ---
# jupyter:
#   jupytext:
#     comment_magics: false
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown] colab_type="text" id="A--sgCKzn9oD"
# # Logistic Regression

# %% [markdown] colab_type="text" id="KFBDBTV_n9oE"
# ## Simple input
#
# check 0x04 for a dataloader version

# %% colab={} colab_type="code" id="v5Gd8ez6n9oF"
from torch import tensor
from torch import nn
from torch import sigmoid
import torch.nn.functional as F
import torch.optim as optim

# %% colab={} colab_type="code" id="giA9i3Unn9oM"
# Training data and ground truth
x_data = tensor([[1.0], [2.0], [3.0], [4.0]])
y_data = tensor([[0.], [0.], [1.], [1.]])


# %% colab={} colab_type="code" id="uEytgyTRn9oS"
class Model_single(nn.Module):
    def __init__(self):
        """
        In the constructor we instantiate nn.Linear module
        """
        super(Model_single, self).__init__()
        self.linear = nn.Linear(1, 1)  # One in and one out

    def forward(self, x):
        """
        In the forward function we accept a Variable of input data and we must return
        a Variable of output data.
        """
        # softmax function is an extension of the sigmoid function to the multiclass case
        y_pred = sigmoid(self.linear(x))
        return y_pred


# %% colab={} colab_type="code" id="ZWMZ2Zl-n9oW" outputId="3e22e72b-4a95-465b-e87d-48410d2cdbba"
# our model
model = Model_single()

# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the model.
# !! For BCEWithLogitsLoss, the last layer of your network should be a Linear
# Sigmoid followed by BCELoss is mathematically equivalent to BCEWithLogitsLoss
# CrossEntropyLoss (which would better be called 'CategoricalCrossEntropyWithLogitsLoss') is essentially the same as BCEWithLogitsLoss
# but requires making some small modifications to your network and your ground-truth labels that add a small amount of unnecessary redundancy to your network
criterion = nn.BCELoss(reduction='mean')
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop
for epoch in range(1000):
    # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x_data)

    # Compute and print loss
    loss = criterion(y_pred, y_data)
    if epoch % 100 == 0:
        print(f'Epoch {epoch + 1}/1000 | Loss: {loss.item():.4f}')

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# %% colab={} colab_type="code" id="a_XD8DN2n9ob" outputId="f3e361f3-85cf-4093-bc66-c42bd4a8e8b9"
# After training
print(f'\nLet\'s predict the hours need to score above 50%\n{"=" * 50}')
hour_var = model(tensor([[1.0]]))
print(
    f'Prediction after 1 hour of training: {hour_var.item():.4f} | Above 50%: {hour_var.item() > 0.5}')
hour_var = model(tensor([[7.0]]))
print(
    f'Prediction after 7 hours of training: {hour_var.item():.4f} | Above 50%: { hour_var.item() > 0.5}')

# %% [markdown] colab_type="text" id="Q5xDtH1Xn9oh"
# ## Multiple inputs
#
# https://pytorch.org/docs/stable/nn.html?highlight=bceloss#torch.nn.BCELoss
#
# https://en.wikipedia.org/wiki/Cross_entropy

# %% colab={} colab_type="code" id="B4r9yxl8n9oi"
from torch import nn, optim, from_numpy
import numpy as np

# %% colab={} colab_type="code" id="NvzmIJMPn9om" outputId="29124f9b-1575-43a4-fb9d-d700267bced0"
data = np.loadtxt('./data/diabetes.csv.gz', delimiter=',', dtype=np.float32)
# row, col
x_data = from_numpy(data[:, 0:-1])
y_data = from_numpy(data[:, [-1]])  # 0 or 1
print(f'X\'s shape: {x_data.shape} | Y\'s shape: {y_data.shape}')


# %% colab={} colab_type="code" id="40V2q7jCn9oq"
# careful with gradient vanishing or explosion
# just a deeper model with multiple inputs
class Model(nn.Module):
    def __init__(self):
        """
        In the constructor we instantiate two nn.Linear module
        """
        super(Model, self).__init__()
        self.l1 = nn.Linear(8, 6)
        self.l2 = nn.Linear(6, 4)
        self.l3 = nn.Linear(4, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        In the forward function we accept a Variable of input data and we must return
        a Variable of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Variables.
        """
        out1 = self.sigmoid(self.l1(x))
        out2 = self.sigmoid(self.l2(out1))
        y_pred = self.sigmoid(self.l3(out2))
        return y_pred


# %% colab={} colab_type="code" id="Vn3x_glLn9ou"
# our model
model = Model()


# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the model.
criterion = nn.BCELoss(reduction='mean')  # binary cross entropy
optimizer = optim.SGD(model.parameters(), lr=0.1)

# %% colab={} colab_type="code" id="K6bo5wF7n9oy" outputId="ae9f4149-6e66-4d0b-b09e-1736cc76aa1c"
# Training loop
for epoch in range(100):
    # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x_data)

    # Compute and print loss
    loss = criterion(y_pred, y_data)
    if epoch % 10 == 0:
        print(f'Epoch: {epoch + 1}/100 | Loss: {loss.item():.4f}')

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# %% colab={} colab_type="code" id="8jjyoOj3n9o2" outputId="96d3f902-0772-4dbf-fc4a-a11071e8ca0d"
# Since I don't have additional data to text...
var = model(x_data)
print((y_data - var).mean())
