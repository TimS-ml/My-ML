# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.10.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown] id="giHLbvSllx1V"
# # Difference between Pytorch and Tensorflow
#
# https://towardsdatascience.com/pytorch-vs-tensorflow-in-code-ada936fd5406
#
# http://cs231n.stanford.edu/slides/2019/cs231n_2019_lecture07.pdf
#
# http://cs231n.stanford.edu/slides/2019/cs231n_2019_lecture08.pdf

# %% [markdown] id="iLeHZBpjx4Gz"
# ## Preprocessing

# %% id="z948Ddpqvb5p"
import io
import requests
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer

url = 'https://raw.githubusercontent.com/TimS-ml/DataMining/master/z_Other/tweets.csv'

f = requests.get(url).content
df = pd.read_csv(io.StringIO(f.decode('utf-8')))
df = df.iloc[:, 1:]
df.columns = ['sentiments', 'tweets']

# df.shape  # (31962, 2)
df.head()

# %% id="v2LXiSccy8tl"
# instantiate and fit tokenizer
tokenizer = Tokenizer(num_words=20000, oov_token='<00v>')
tokenizer.fit_on_texts(df.tweets)

# transform tweets into sequences of integers
sequences = tokenizer.texts_to_sequences(df.tweets)

# pad sequences so that they have uniform lenth
padded = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=42)
assert(padded.shape==(31962, 42))

seq = padded
labels = np.array(df.sentiments)

# %% [markdown] id="1b7ik9-voHXm"
# # Pytorch
#
# There are two ways to build a neural network model in PyTorch.
#
#

# %% [markdown] id="1HANalHbpm5p"
# ## Two ways of building NN in PT

# %% id="Bj-1R6zDtJ-N"
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# %% [markdown] id="3KCSLl3eo6vw"
# ### [1] Model Subclassing
# Similar to TensorFlow, in PyTorch you subclass the `nn.Model` module and define your layers in the `__init__()` method. 
#
# The only difference is that you create the `forward` pass in a method named forward *instead of `call`*.
#
# Difference to the Keras model: <u>There’s only an average-pooling layer in PyTorch so it needs to have the right kernel size in order the make it global average-pooling.</u>

# %% id="V5LxP3IZoskt"
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.embedding_layer = nn.Embedding(num_embeddings=20000,
                                            embedding_dim=50)
        self.pooling_layer = nn.AvgPool1d(kernel_size=50)
        self.fc_layer = nn.Linear(in_features=42, out_features=1)
    
    def forward(self, inputs):
        x = self.embedding_layer(inputs)
        x = self.pooling_layer(x).view(32, 42)
        return torch.sigmoid(self.fc_layer(x))
    
model = Model()

# %% [markdown] id="KQwfyIHYo02W"
# ### [2] Sequential
# PyTorch also offers a `Sequential` module that looks almost equivalent to TensorFlow’s.
#
# Many layers do not work with PyTorch’s `nn.Sequential`

# %% id="P110pyzxp3Q2"
# PyTorch nn.Sequential
model = nn.Sequential(
    nn.Embedding(num_embeddings=20000, embedding_dim=50),
    nn.AvgPool1d(kernel_size=50),
    nn.Flatten(start_dim=1),
    nn.Linear(in_features=42, out_features=1),
    nn.Sigmoid()
)

# %% [markdown] id="w3OoBbFDz1LN"
# ## Training a NN in PT
#
# https://pytorch.org/tutorials/beginner/pytorch_with_examples.html
#
# Training loop needs to be implemented from scratch
#
# In oder to process the data in batches, a dataloader must be created. The dataloader returns one batch at a time in a dictionary format.
#
# Short description of the training loop: 
# - For each batch, we calculate the loss and then call loss.backward() to backpropagate the gradient through the layers. 
# - In addition, we call optimizer.step() to tell the optimizer to update the parameters. 
#

# %% id="mOrmaWNVqdMS"
# define the loss fn and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# initialize empty list to track batch losses
batch_losses = []

# train the neural network for 5 epochs
for epoch in range(5):
    # reset iterator
    dataiter = iter(dataloader)
    
    for batch in dataiter:
        # reset gradients
        optimizer.zero_grad()
        
        # forward propagation through the network
        out = model(batch["tweets"])
        
        # calculate the loss
        loss = criterion(out, batch["sentiments"])
        
        # track batch loss
        batch_losses.append(loss.item())
        
        # backpropagation
        loss.backward()
        
        # update the parameters
        optimizer.step()

# %% [markdown] id="Z7_EnL-lnHSN"
# # Tensorflow
#
# TensorFlow is a lot like Scikit-Learn thanks to its `fit` function, which makes training a model super easy and quick.
#
# There are three ways to build a neural network model in Keras.

# %% [markdown] id="Xu-Cfjwtq7h6"
# ## Three ways of building NN in TF

# %% id="THDUGfhivjlq"
import tensorflow as tf


# %% [markdown] id="ygwFVFKDrG-H"
#
# ### [1] Model subclassing
#
# You can create your own fully-customizable models by subclassing the `tf.keras.Model` class and implementing the forward pass in the `call` method. 
#
# Put differently, layers are defined in the __init__() method and the logic of the forward pass in the call method.
#

# %% id="6Ek17JFhrPRu"
class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        self.embedding_layer = tf.keras.layers.Embedding(input_dim=20000,
                                                         output_dimension=50,
                                                         input_length=42,
                                                         mask_zero=True)
        self.flatten_layer = tf.keras.layers.Flatten()
        self.fc1_layer =  tf.keras.layers.Dense(128, activation='relu')
        self.fc2_layer =  tf.keras.layers.Dense(1, activation='sigmoid')
        
    def call(self, inputs):
        x = self.embedding_layer(inputs)
        x = self.flatten_layer(x)
        x = self.fc1_layer(x)
        return self.fc2_layer(x)
        
model = Model()

# %% [markdown] id="LJrdFWoSrUO-"
#
# ### [2] Functional API
# Given some input tensor(s) and output tensor(s), you can also instantiate 实例化 a `Model`. 
#
# With this approach, you essentially define a layer and immediately pass it the input of the previous layer. 
#

# %% id="hDF4d0uDra1D"
inputs = tf.keras.layers.Input(shape=(42,))
x = tf.keras.layers.Embedding(input_dim=20000,
                              output_dimension=50,
                              input_length=42,
                              mask_zero=True)(inputs)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(128, activation='relu')(x)
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

# %% [markdown] id="VsZiUiOergva"
# ### [3] Sequential model API
# Typically consisting of just a few common layers — kind of a shortcut to a trainable model. 
#
# Too inflexible if you wish to implement more sophisticated ideas.
#
#

# %% id="-2M2t9CerGam"
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=20000,
                              output_dimension=50,
                              input_length=42,
                              mask_zero=True),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')])

# %% [markdown] id="EDFCfDxvwzUq"
# ## Two useful functions of TF
#
# First, calling `model.summary`() prints a compact summary of the model and the number of parameters
#
# Second, by calling `tf.keras.utils.plot_model()` you get a graphical summary of the model.

# %% [markdown] id="5GfIbncgxM9-"
# ## Training a NN in Keras
#
# Before you can train a Keras model, it must be compiled by running the `model.compile()` function, which is also where you specify the loss function and optimizer.
#
# ```python
# model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
# ```
#
# Keras models have a convenient `model.fit()` function for training a model (just like Scikit-Learn), which also takes care of batch processing and even evaluates the model on the run (if you tell it to do so).
#
# ```python
# model.fit(x=X, y, batch_size=32, epochs=5, verbose=2, validation_split=0.2)
# ```
