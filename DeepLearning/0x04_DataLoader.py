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

# %% [markdown] toc-hr-collapsed=true id="1k_JNswbNNsk" colab_type="text"
# # DataLoader
#
# https://pytorch.org/docs/stable/data.html?highlight=dataloader#torch.utils.data.DataLoader
#
# https://pytorch.org/docs/stable/data.html?highlight=dataset#torch.utils.data.Dataset
#

# %% [markdown] id="eizn2kH6NNsm" colab_type="text"
# ## go through dataset
#
# basically is a data viewer
#
# wrap in tensor in each epoach

# %% id="1_3v0lB8NNsn" colab_type="code" colab={}
# References
# https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/01-basics/pytorch_basics/main.py
# http://pytorch.org/tutorials/beginner/data_loading_tutorial.html#dataset-class
from torch.utils.data import Dataset, DataLoader
from torch import from_numpy, tensor
import numpy as np


# %% id="HhsjNrsqNNst" colab_type="code" colab={}
class DiabetesDataset(Dataset):  # Dataset is from torch too
    """ Diabetes dataset."""

    # Initialize your data, download, etc.
    def __init__(self):
        xy = np.loadtxt('./data/diabetes.csv.gz',
                        delimiter=',', dtype=np.float32)
        self.len = xy.shape[0]
        self.x_data = from_numpy(xy[:, 0:-1])
        self.y_data = from_numpy(xy[:, [-1]])

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


# %% id="1mEIOjIoNNsy" colab_type="code" colab={}
dataset = DiabetesDataset()
train_loader = DataLoader(dataset=dataset,
                          batch_size=32,
                          shuffle=True,
                          num_workers=2)  # how many subprocesses to use for data loading

# %% id="qxRL-xloNNs3" colab_type="code" colab={} outputId="f9963b35-16af-4ebe-dfce-cdc08ec26e10"
# nothing interesting, just go over the data 2 times
for epoch in range(2):
    for i, data in enumerate(train_loader, 0):
        # get the inputs
        inputs, labels = data
        
        # wrap them in Variable
        inputs, labels = tensor(inputs), tensor(labels)
        
        # Run your training process
        if i % 5 == 0:
            print(f'Epoch: {i}')
            # print(f'Inputs {inputs.data} | Labels {labels.data}')

# %% [markdown] id="V5wBz8ICNNs-" colab_type="text"
# ## Logistic regression with data loader
#
# https://pytorch.org/docs/stable/torch.html?highlight=from_numpy#torch.from_numpy
#
# https://pytorch.org/docs/stable/optim.html?highlight=optim%20sgd#torch.optim.SGD
#
# https://pytorch.org/docs/stable/nn.html?highlight=nn%20bceloss#torch.nn.BCELoss
#

# %% id="X42WIHlZNNs_" colab_type="code" colab={}
# References
# https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/01-basics/pytorch_basics/main.py
# http://pytorch.org/tutorials/beginner/data_loading_tutorial.html#dataset-class
from torch import nn, optim


# %% id="uSMwrmn4NNtE" colab_type="code" colab={}
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


# %% id="qw90UlySNNtJ" colab_type="code" colab={}
# our model
model = Model()

# %% id="pLgW1ZUTNNtO" colab_type="code" colab={} outputId="b1b876d4-999a-433a-817a-19e43989b1b1"
# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the model.
criterion = nn.BCELoss(reduction='sum')
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Training loop
for epoch in range(2):
    # enumerate(iter, start)
    for i, data in enumerate(train_loader):
        # get the inputs
        inputs, labels = data

        # Forward pass: Compute predicted y by passing x to the model
        y_pred = model(inputs)

        # Compute and print loss
        loss = criterion(y_pred, labels)
        if (i+1) % 6 == 0:
            print(f'Epoch {epoch + 1} | Batch: {i+1} | Loss: {loss.item():.4f}')

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
