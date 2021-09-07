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
#     display_name: Python (torch)
#     language: python
#     name: torch
# ---

# %% [markdown]
# # RNN

# %%
import sys
import torch
import torch.nn as nn
from torch.autograd import Variable

# %%
torch.manual_seed(777)  # reproducibility

# %% [markdown]
# ## Basic RNN cell

# %%
# One hot encoding for each char in 'hello'
h = [1, 0, 0, 0]
e = [0, 1, 0, 0]
l = [0, 0, 1, 0]
o = [0, 0, 0, 1]

# %%
# One cell RNN input_dim (4) -> output_dim (2). sequence: 5
cell = nn.RNN(input_size=4, hidden_size=2, batch_first=True)

# %% [markdown]
# ### Single batch

# %%
# (num_layers * num_directions, batch, hidden_size) 
# whether batch_first=True or False
hidden = Variable(torch.randn(1, 1, 2))
inputs = Variable(torch.Tensor([h, e, l, l, o]))

for one in inputs:
    one = one.view(1, 1, -1)
    # Input: (batch, seq_len, input_size) when batch_first=True
    out, hidden = cell(one, hidden)
    print("one input size", one.size(), "out size", out.size())

# %%
# We can do the whole at once

# Propagate input through RNN
# Input: (batch, seq_len, input_size) when batch_first=True
inputs = inputs.view(1, 5, -1)
out, hidden = cell(inputs, hidden)
print("sequence input size", inputs.size(), "out size", out.size())

# %% [markdown]
# ### Multiple batches

# %%
hidden = Variable(torch.randn(1, 3, 2))

# One cell RNN input_dim (4) -> output_dim (2). sequence: 5, batch 3
# 3 batches 'hello', 'eolll', 'lleel'
# rank = (3, 5, 4)
inputs = Variable(
    torch.Tensor([[h, e, l, l, o], [e, o, l, l, l], [l, l, e, e, l]]))

# Propagate input through RNN
# Input: (batch, seq_len, input_size) when batch_first=True
# B x S x I
out, hidden = cell(inputs, hidden)
print("batch input size", inputs.size(), "out size", out.size())

# %% [markdown]
# ### Remove batch first

# %%
# One cell RNN input_dim (4) -> output_dim (2)
cell = nn.RNN(input_size=4, hidden_size=2)

# The given dimensions dim0 and dim1 are [swapped].
inputs = inputs.transpose(dim0=0, dim1=1)
# Propagate input through RNN
# Input: (seq_len, batch_size, input_size) when batch_first=False (default)
# S x B x I
out, hidden = cell(inputs, hidden)
print("batch input size", inputs.size(), "out size", out.size())

# %% [markdown]
# ## RNN
#
# no dataloader yet...
#
# Teach hihell -> ihello

# %%
# char: h; idx: 0; onehot: 1, 0, 0 ... 0
#            0    1    2    3    4
idx2char = ['h', 'i', 'e', 'l', 'o']

# Teach hihell -> ihello
x_data = [0, 1, 0, 2, 3, 3]  # hihell
y_data = [1, 0, 2, 3, 3, 4]  # ihello

one_hot_lookup = [
    [1, 0, 0, 0, 0],  # 0
    [0, 1, 0, 0, 0],  # 1
    [0, 0, 1, 0, 0],  # 2
    [0, 0, 0, 1, 0],  # 3
    [0, 0, 0, 0, 1]   # 4
]

x_one_hot = [one_hot_lookup[x] for x in x_data]

# As we have one batch of samples, we will change them to variables only once
inputs = Variable(torch.Tensor(x_one_hot))
labels = Variable(torch.LongTensor(y_data))

# %%
num_classes = 5
input_size = 5  # one-hot size
hidden_size = 5  # output from the RNN. 5 to directly predict one-hot
batch_size = 1  # one sentence
sequence_length = 1  # One by one
num_layers = 1  # one-layer rnn


# %% [markdown]
# ### Model

# %%
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.rnn = nn.RNN(input_size=input_size,
                          hidden_size=hidden_size,
                          batch_first=True)

    def forward(self, hidden, x):
        # Reshape input (batch first)
        # x: tensor([1., 0., 0., 0., 0.])
        # it will add more dim to x
        x = x.view(batch_size, sequence_length, input_size)

        # Propagate input through RNN
        # Input: (batch, seq_len, input_size)
        # hidden: (num_layers * num_directions, batch, hidden_size)
        out, hidden = self.rnn(x, hidden)
        return hidden, out.view(-1, num_classes)

    def init_hidden(self):
        # Initialize hidden and cell states
        # (num_layers * num_directions, batch, hidden_size)
        return Variable(torch.zeros(num_layers, batch_size, hidden_size))


# %%
# Instantiate RNN model
model = Model()
print(model)

# Set loss and optimizer function
# CrossEntropyLoss = LogSoftmax + NLLLoss
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

# Train the model
for epoch in range(100):
    optimizer.zero_grad()
    loss = 0
    hidden = model.init_hidden()

    for data_in, label in zip(inputs, labels):
        # print shape of inputs
        # print(data_in, label)
        # torch.Size([1, 1, 5]) torch.Size([1, 5])
        hidden, output = model(hidden, data_in)

        _, idx = output.max(1)
        if epoch % 10 == 0:
            print('predicted string: ', 
                  idx2char[idx.data[0]])  # visualize char in output
            
        # RuntimeError: dimension specified as 0 but tensor has no dimensions
        # loss += criterion(output, label)
        loss += criterion(output, torch.LongTensor([label]))

    if epoch % 10 == 0:
        print("epoch: %d, loss: %1.3f" % (epoch + 1, loss))

    loss.backward()
    optimizer.step()

print("Learning finished!")

# %% [markdown]
# ## RNN in Seq

# %%
inputs = Variable(torch.Tensor([x_one_hot]))  # why?

sequence_length = 6  # len(ihello) == 6


# %% [markdown]
# ### Model

# %%
class RNN(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super(RNN, self).__init__()

        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length

        self.rnn = nn.RNN(input_size=self.input_size, 
                          hidden_size=self.hidden_size, 
                          batch_first=True)

    def forward(self, x):
        # Initialize hidden and cell states
        # (num_layers * num_directions, batch, hidden_size) for batch_first=True
        h_0 = Variable(
            torch.zeros(self.num_layers, x.size(0), self.hidden_size))

        # Reshape input: batch size, seq len, input size
        x.view(x.size(0), self.sequence_length, self.input_size)

        # Propagate input through RNN
        # Input: (batch, seq_len, input_size)
        # h_0: (num_layers * num_directions, batch, hidden_size)

        out, _ = self.rnn(x, h_0)
        return out.view(-1, num_classes)


# %%
# Instantiate RNN model
rnn = RNN(num_classes, input_size, hidden_size, num_layers)
print(rnn)

# Set loss and optimizer function
# CrossEntropyLoss = LogSoftmax + NLLLoss
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(rnn.parameters(), lr=0.1)

# Train the model
for epoch in range(100):
    outputs = rnn(inputs)
    optimizer.zero_grad()
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    _, idx = outputs.max(1)
    idx = idx.data.numpy()
    result_str = [idx2char[c] for c in idx.squeeze()]
    if epoch % 10 == 0:
        print("epoch: %d, loss: %1.3f" % (epoch + 1, loss.data))
        print("Predicted string: ", ''.join(result_str))

print("Learning finished!")

# %% [markdown]
# ## Use embedding

# %%
x_data = [[0, 1, 0, 2, 3, 3]]  # hihell

inputs = Variable(torch.LongTensor(x_data))

# %%
embedding_size = 10  # embedding size


# %% [markdown]
# ### Model

# %%
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # this is new
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.RNN(input_size=embedding_size,
                          hidden_size=5,
                          batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Initialize hidden and cell states
        # (num_layers * num_directions, batch, hidden_size)
        h_0 = Variable(
            torch.zeros(num_layers, x.size(0), hidden_size))

        emb = self.embedding(x)
        emb = emb.view(batch_size, sequence_length, -1)

        # Propagate embedding through RNN
        # Input: (batch, seq_len, embedding_size)
        # h_0: (num_layers * num_directions, batch, hidden_size)
        out, _ = self.rnn(emb, h_0)
        return self.fc(out.view(-1, num_classes))


# %%
# Instantiate RNN model
model = Model()
print(model)

# Set loss and optimizer function
# CrossEntropyLoss = LogSoftmax + NLLLoss
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

# Train the model
for epoch in range(100):
    outputs = model(inputs)
    optimizer.zero_grad()
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    _, idx = outputs.max(1)
    idx = idx.data.numpy()
    result_str = [idx2char[c] for c in idx.squeeze()]
    if epoch % 10 == 0:
        print("epoch: %d, loss: %1.3f" % (epoch + 1, loss.data))
        print("Predicted string: ", ''.join(result_str))

print("Learning finished!")
