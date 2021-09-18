# ---
# jupyter:
#   jupytext:
#     comment_magics: true
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.12.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown] id="pXk3OuG6m2PX"
# https://github.com/eriklindernoren/ML-From-Scratch/blob/master/mlfromscratch/deep_learning/activation_functions.py
#
# https://github.com/eriklindernoren/ML-From-Scratch/blob/master/mlfromscratch/deep_learning/loss_functions.py

# %% id="sWOZrPyilLWh"
import numpy as np

import matplotlib.pyplot as plt

# %matplotlib inline


# %% id="2wauo23fZfwh"
def set_default(figsize=(10, 10), dpi=100):
    plt.style.use(['dark_background', 'bmh'])
    plt.rc('axes', facecolor='k')
    plt.rc('figure', facecolor='k')
    plt.rc('figure', figsize=figsize, dpi=dpi)
    plt.rc('font', size=15)


set_default()

# %% [markdown] id="WK92yxdEnDfE"
# # Activation Function

# %% [markdown] id="mMkFLijIlLWn"
# ## Tanh


# %% id="DkrLM8QslLWp" colab={"base_uri": "https://localhost:8080/", "height": 832} outputId="b0d026d9-5c63-4d02-b750-eac896b97d73"
def tanh(x):
    return 2 / (1 + np.exp(-2 * x)) - 1


a = np.arange(-5, 5, 0.01)
plt.plot(a, tanh(a))
plt.xlabel("$x$")
plt.ylabel("$Tanh(x)$")

# %% id="fsxwiS2elLWq"
# (a[501] - a[500]) / 0.01

# %% [markdown] id="0BvHdSiBlreW"
# ## Sigmoid


# %% id="Tw4Yr78nlLWr" colab={"base_uri": "https://localhost:8080/", "height": 845} outputId="9473c2cb-beb5-4aca-e9e3-25e9a81add96"
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


a = np.arange(-5, 5, 0.01)
plt.plot(a, sigmoid(a))
plt.xlabel("$x$")
plt.ylabel("$Sigmoid(x)$")

# %% [markdown] id="j2XXxusRYEu8"
# ### Derivates of Sigmoid vs Tanh
#
# it produces relatively flat gradients during the backward pass

# %% id="jnx9PtDwlLWs" colab={"base_uri": "https://localhost:8080/", "height": 881} outputId="553ffa20-241d-451f-ff72-2f056475f380"
a = np.arange(-5, 5, 0.01)

plt.plot(a, sigmoid(a) * (1 - (sigmoid(a))))
plt.plot(a, 1 - (np.tanh(a)**2))
plt.legend(['Derivative of $sigmoid(x)$', 'Derivative of $Tanh(x)$'])
plt.title("Derivatives of $sigmoid(x)$ and $Tanh(x)$")
plt.xlabel("x")

# %% [markdown] id="lhqyIXLXlK3g"
# ## Softmax
#
# Input is logits:
# - Output FC (with/without activation)
#
# Output is Prob, and sum of the output seqence is 1


# %% colab={"base_uri": "https://localhost:8080/", "height": 835} id="XbObowx4lMK0" outputId="178c4f1b-546a-48c2-e4d9-92eb6fe700ce"
def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)


a = np.arange(-5, 5, 0.01)
plt.plot(a, softmax(a))
plt.xlabel("$x$")
plt.ylabel("$Softmax(x)$")

# %% colab={"base_uri": "https://localhost:8080/"} id="oCvhWnaEvl31" outputId="8d6b0021-6158-4c5f-95fd-8908d724e728"
x1 = np.array([1, 2, 3])
x2 = np.array([1, 2, 5])

print(softmax(x1), softmax(x2))

# %% [markdown] id="6Q7TmVlNltOs"
# ## ReLU


# %% id="9jikx03SlLWs" colab={"base_uri": "https://localhost:8080/", "height": 845} outputId="fe6748f9-2c1c-4fe4-d2ae-26feeb5c771a"
def relu(x):
    # return np.array([el if el > 0 else 0 for el in x])
    return np.where(x >= 0, x, 0)


a = np.arange(-5, 5, 0.01)
plt.plot(a, relu(a))
plt.xlabel("X")
plt.ylabel("$ReLU(x)$")

# %% [markdown] id="RHeK9pasZSf5"
# ## LeakyReLU


# %% colab={"base_uri": "https://localhost:8080/", "height": 845} id="GSgQ4Q7DZXYm" outputId="422c4bbf-f127-49e5-84d4-4058a28e91cf"
def leaky_relu(x, alpha=0.2):
    return np.where(x >= 0, x, alpha * x)


a = np.arange(-5, 5, 0.01)
plt.plot(a, leaky_relu(a))
plt.xlabel("X")
plt.ylabel("$LeakyReLU(x)$")

# %% [markdown] id="aOwhhnT6lLWu"
# # Cross Entropy + Softmax
#
# SCE = Softmax Cross Entropy

# %% [markdown] id="FoyhrIABlLWu"
# $$ \text{CE}(p_i, y_i) = - y_i * \text{log}(p_i) - (1 - y_i) * \text{log}(1-p_i) $$

# %% [markdown] id="bDFP0iq9lLWv"
# $$ CE(x, 0) = \sum_{i}{(- y_i * log(p_i) - (1 - y_i) * log(1-p_i))} $$
#
# $$ \begin{align} = &\sum_{i}{(- 0 * log(p_i) - (1 - 0) * log(1-p_i))} \\
# =  &\sum_{i}{- log(1-p_i)} \end{align}$$

# %% [markdown] id="pjcTMZEvlLWv"
# $$ CE(x, 1) = \sum_{i}{(- y_i * log(p_i) - (1 - y_i) * log(1-p_i))} $$
#
# $$ \begin{align} = &\sum_{i}{(- 1 * log(p_i) - (1 - 1) * log(1-p_i))} \\
# =  &\sum_{i}{- log(p_i)} \end{align}$$

# %% [markdown] id="NHPosRFzlLWv"
# $$ \text{Normalize}(\begin{bmatrix} x_1 \\ x_2 \\ x_3 \end{bmatrix})  = \begin{bmatrix} \frac{x_1}{x_1 + x_2 + x_3} \\
# \frac{x_2}{x_1 + x_2 + x_3} \\
# \frac{x_3}{x_1 + x_2 + x_3}
# \end{bmatrix} $$

# %% [markdown] id="Ce4R9swLlLWw"
# $$ \text{Softmax}(\begin{bmatrix} x_1 \\ x_2 \\ x_3 \end{bmatrix})  = \begin{bmatrix} \frac{e^{x_1}}{e^{x_1} + e^{x_2} + e^{x_3}} \\
# \frac{e^{x_2}}{e^{x_1} + e^{x_2} + e^{x_3}} \\
# \frac{e^{x_3}}{e^{x_1} + e^{x_2} + e^{x_3}}
# \end{bmatrix} $$

# %% [markdown] id="L3TUygqTlLWw"
# $$ S(\begin{bmatrix} x_1 \\ x_2 \\ x_3 \\ \vdots \\ x_n \end{bmatrix})  = \begin{bmatrix} \frac{e^{x_1}}{e^{x_1} + e^{x_2} + e^{x_3} + \ldots + e^{x_n}} \\
# \frac{e^{x_2}}{e^{x_1} + e^{x_2} + e^{x_3} + \ldots + e^{x_n}} \\
# \frac{e^{x_3}}{e^{x_1} + e^{x_2} + e^{x_3} + \ldots + e^{x_n}} \\
# \ldots \\
# \frac{e^{x_n}}{e^{x_1} + e^{x_2} + e^{x_3} + \ldots + e^{x_n}}
# \end{bmatrix} $$

# %% [markdown] id="-KWqNTxslLWw"
# $$ {SCE}_1 = - y_1 * log(\frac{e^{x_1}}{e^{x_1} + e^{x_2} + e^{x_3}}) - (1 - y_1) * log(1-\frac{e^{x_1}}{e^{x_1} + e^{x_2} + e^{x_3}}) $$

# %% [markdown] id="5uiTDZjZlLWx"
# $$ \text{softmax}(\begin{bmatrix} p_1 \\ p_2 \\ p_3 \end{bmatrix}) - \begin{bmatrix} y_1 \\ y_2 \\ y_3 \end{bmatrix} $$

# %% [markdown] id="TY3nuKfQ2VT_"
# ## vs MSE

# %% [markdown] id="tLJECJOjlLWx"
# ### Plot of log loss when $y_i = 0$

# %% id="YwZxk25clLWy" colab={"base_uri": "https://localhost:8080/", "height": 883} outputId="50a7c66b-57b1-413b-f45f-4fa50ee34087"
x = np.linspace(0.01, 0.99, 99)
y1 = -1.0 * np.log(1 - x)
y2 = (x - 0)**2

plt.plot(x, y1)
plt.plot(x, y2)
plt.legend(['Cross entropy loss', 'Mean squared error'])

plt.title("Cross entropy loss vs. MSE when $y = 0$")
plt.xlabel("Prediction ($p$)")
plt.ylabel("Loss values")
# plt.savefig(IMG_FILEPATH + "04_Log_loss_vs_MSE_y_eq_0.png")

# %% [markdown] id="4iZW61QblLWz"
# When $y=0$, loss can become theoretically infinite as $p$ approaches 1.

# %% [markdown] id="5jyzov0zlLWz"
# ### Plot of loss when $y_i = 1$

# %% id="ESVnhDz7lLWz" colab={"base_uri": "https://localhost:8080/", "height": 883} outputId="2e95f56a-bfa0-4617-e247-bf3a568f9de7"
x = np.linspace(0.01, 0.99, 99)
y = -1.0 * np.log((x))

plt.plot(x, y)
plt.title("Log loss for $y = 1$")
plt.xlabel("Prediction ($p$)")
plt.ylabel("Log loss")

# %% [markdown] id="S8ln6qaMlLW0"
# ## Derivation of Softmax Cross Entropy derivative (simple case, two probabilities)

# %% [markdown] id="uJtVPgGllLW0"
# $$ C(p, y) = - y * \text{log}(p) - (1 - y) * \text{log}(1-p) $$

# %% [markdown] id="7f_F2H-MlLW0"
# $$
# C(p,y)=
# \begin{cases}
# -log(1-p) & \text{if }  y = 0\\
# -log(p) & \text{if }  y = 1
# \end{cases}
# $$

# %% [markdown] id="OS3mxB8GlLW0"
# $$ \text{SCE}(\begin{bmatrix} p_1 \\ p_2 \end{bmatrix}, \begin{bmatrix} y_1 \\ y_2 \end{bmatrix})_1 = - y_1 * log(S(\begin{bmatrix} p_1 \\ p_2 \end{bmatrix})_1) - (1 - y_1) * log(1-S(\begin{bmatrix} p_1 \\ p_2 \end{bmatrix})_1) = - y_1 * log(\frac{e^{p_1}}{e^{p_1} + e^{p_2}}) - (1 - y_1) * log(1-\frac{e^{p_1}}{e^{p_1} + e^{p_2}}) $$

# %% [markdown] id="5Hf3cIPZlLW1"
# Structure:
#
# $$ y_1 = a $$
# $$e^{x_2} = b$$
#
# $$ SC(x_1) = - a * log(\frac{e^{x_1}}{e^{x_1} + b}) - (1 - a) * log(1-\frac{e^{x_1}}{e^{x_1} + b}) $$
#
# **Quotient rule:**
#
# $$ f(x) = \frac{g(x)}{h(x)} $$
#
# $$ f'(x) = \frac{g'(x) * h(x) - g(x) * h'(x)}{(h(x))^2} $$

# %% [markdown] id="nSHmq2GblLW1"
# If
#
# $$ f(x) = \frac{e^x}{e^x + b} $$
#
# $$ \begin{align} f'(x) =& \frac{e^x * (e^x + b) - (e^x * e^x)}{(e^x + b)^2} \\
# =& \frac{e^x * (e^x + b - e^ x)}{(e^x + b)^2} \\
# =& \frac{e^x * b}{(e^x + b)^2}\end{align} $$

# %% [markdown] id="yd5Y8BbRlLW1"
# And if:
#
# $$ g(x) = - a * log(f(x)) - (1 - a) * log(1-f(x)) $$
#
# then:
#
# $$ g'(x) = - a * \frac{f'(x)}{f(x)} - (1 - a) * \frac{-1 * f'(x)}{1-f(x)} $$
#
# First, we'll compute $\frac{f'(x)}{f(x)}$:
#
# $$ \begin{align}
# \frac{f'(x)}{f(x)} =& \frac{\frac{e^x * b}{(e^x + b)^2}}{\frac{e^x}{e^x + b}} \\\\
# =& \frac{-e^x * b}{(e^x + b)^2} * \frac{e^x + b}{e^x}
#  \\
# =& \frac{b}{e^x + b} \end{align} $$

# %% [markdown] id="kqrgjcbKlLW1"
# Now, in this next part, we'll use the fact that:
#
# $$ \frac{b}{e^x + b} = 1 - \frac{e^x}{e^x + b} $$

# %% [markdown] id="lPmSOjrMlLW1"
# $$ \begin{align}
# \frac{-1 * f'(x)}{1 - f(x)} =& \frac{-1 * \frac{e^x * b}{(e^x + b)^2}}{1 - \frac{e^x}{e^x + b}} \\
# =& \frac{\frac{-e^x * b}{(e^x + b)^2}}{\frac{b}{e^x + b}} \\
# =& \frac{-e^x * b}{(e^x + b)^2} * \frac{e^x + b}{b}
#  \\
# =& \frac{-e^x}{e^x + b} \end{align}$$

# %% [markdown] id="LiBdkYm_lLW1"
# Finally, putting these pieces together:
#
# $$ \begin{align} SC'(x) =& - a * \frac{f'(x)}{f(x)} - (1 - a) * \frac{-1 * f'(x)}{1-f(x)} \\
# =& -a * \frac{b}{e^x + b} - (1 - a) * \frac{-e^x}{e^x + b} \\
# =& -a * \frac{b}{e^x + b} + \frac{e^x}{e^x + b} - a * \frac{-e^x}{e^x + b} \\
# =& -a * (1 - \frac{e^x}{e^x + b}) + \frac{e^x}{e^x + b} - a * \frac{-e^x}{e^x + b} \\
# =& -a + a * \frac{e^x}{e^x + b} + \frac{e^x}{e^x + b} - a * \frac{-e^x}{e^x + b} \\
# =& -a + \frac{e^x}{e^x + b} \\
# \end{align} \\
# $$

# %% [markdown] id="OVfTfhjzlLW2"
# That's right, the derivative to be sent backward from the softmax layer is simply:
#
# $$ S - Y = s(\begin{bmatrix} p_1 \\ p_2 \end{bmatrix}) - \begin{bmatrix} y_1 \\ y_2 \end{bmatrix} = \begin{bmatrix} \frac{e^{p_1}}{e^{p_1} + e^{p_2}} - y_1 \\ \frac{e^{p_2}}{e^{p_1} + e^{p_2}} - y_2 \end{bmatrix} $$
#
# This makes sense:
#
# * The softmax output will always be between 0 and 1.
# * If $y_i$ is 0, then $ s(x_1) - y_1 $ will be a positive number: because indeed, if we increase the value of $x_1$, the loss will increase. Conversely if $y_i$ is one.
# * Note that this loss function only makes sense because $ s(x_i) $ is always between 0 and 1.
#
# This, by the way, is why TensorFlow has a function called `softmax_cross_entropy_with_logits`!

# %% [markdown] id="Y2KoH3_hlLW2"
# ## ! Numerically stable softmax math

# %% [markdown] id="Bs49ntZXlLW2"
# $$ log(softmax(x_j)) = log(\frac{e^{x_j}}{\sum_i^n e^{x_j}}) = x_j - logsumexp(X) $$
#
# $$ softmax(x_j) = e^{log(softmax(x_j))} = e^{x - logsumexp(x)} $$
#
# $$ logsumexp(X) = log(\sum_i^n e^{x_i}) = c + log(\sum_i^n e^{x_i - c}) $$

# %% [markdown] id="0-EWHNUTlLW2"
# # One hot encoding
#
# $$ [0, 2, 1] \Rightarrow \begin{bmatrix} 1 & 0 & 0 & \ldots & 0 \\ 0 & 0 & 1 & \ldots & 0 \\ 0 & 1 & 0 & \ldots & 0 \end{bmatrix} $$

# %% [markdown] id="JA6LsuBHlLW2"
# # Learning rate decay

# %% [markdown] id="OktY1GMOlLW2"
# ## Linear decay
#
# $$ \alpha_{t} = \alpha_{start} - (\alpha_{start} - \alpha_{end}) * \frac{t}{N} $$
#
# ## Exponential decay
#
# $$ \alpha_{t} = \alpha_{start} * ((\frac{\alpha_{end}}{\alpha_{start}})^\frac{1}{N})^t $$

# %% [markdown] id="tXNnlL2clLW3"
# # SGD Momentum gradient formula

# %% [markdown] id="NpXkEOnklLW3"
# $$ \text{update} = \nabla_t + \mu * \nabla_{t-1} + \mu^2 * \nabla_{t-2} + \ldots $$

# %% [markdown] id="LsG5CgGnlLW3"
# # Weight init illustrations

# %% [markdown] id="qG2uuztflLW3"
# ## Feature level math

# %% [markdown] id="CVpnFGsulLW3"
# $$ f_n = w_{1, n}* x_1 + \ldots + w_{784, n}* x_{784} + b_n $$

# %% [markdown] id="Oq76y-pjlLW3"
# $$ \text{Var}(w_{i,j}) = 1 $$

# %% [markdown] id="Dq0RSWF_lLW3"
# $$ \text{Var}(X_1 + X_2) = \text{Var}(X_1) + \text{Var}(X_2) $$

# %% id="Shl9BmuClLW4"
n_feat = 784
n_hidden = 256

np.random.seed(190131)

# %% id="_VS1LRD4lLW4"
a = np.random.randn(1, n_feat)

# %% id="Vptl8Qg7lLW4"
b = np.random.randn(n_feat, n_hidden)

# %% id="OVOgVIPBlLW4"
out = np.dot(a, b).reshape(n_hidden)

# %% id="s9fZ5BkalLW4" colab={"base_uri": "https://localhost:8080/", "height": 1000} outputId="6b4eb11d-7906-4353-d28b-1bb95632f6f6"
fig, ax = plt.subplots(2, 1, figsize=(8, 16))
ax[0].hist(out)
ax[0].set_title(
    "Distribution of inputs to activation function\nfor layer with 784 inputs")
ax[1].hist(np.tanh(out))
ax[1].set_title(
    "Distribution of outputs of tanh function\nfor layer with 784 inputs")
# fig.savefig(IMG_FILEPATH + "01_weight_init_activation_illustration.png")
