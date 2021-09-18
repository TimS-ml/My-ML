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

# %% id="ZfYo46JDOK92"
import numpy as np
from numpy import ndarray

import copy

# https://www.geeksforgeeks.org/callable-in-python/
from typing import Callable
from typing import Dict
from typing import List

import matplotlib.pyplot as plt
import matplotlib
# %matplotlib inline

# %% id="2wauo23fZfwh"
def set_default(figsize=(10, 10), dpi=100):
    plt.style.use(['dark_background', 'bmh'])
    plt.rc('axes', facecolor='k')
    plt.rc('figure', facecolor='k')
    plt.rc('figure', figsize=figsize, dpi=dpi)
    plt.rc('font', size=15)


set_default()

# %% [markdown] id="FReSiOCGOK92"
# # Basic Function
#
# https://github.com/TimS-ml/ML-From-Scratch/blob/master/mlfromscratch/deep_learning/activation_functions.py


# %% id="WjzC7VB7OK93"
def square(x: ndarray) -> ndarray:
    return np.power(x, 2)


def leakyRelu(x: ndarray) -> ndarray:
    '''
    Relu with a slight twist
    https://en.wikipedia.org/wiki/Rectifier_(neural_networks)# Leaky_ReLU
    '''
    # relu: x.clip(min=0)
    return np.maximum(.2 * x, x)


def sigmoid(x: ndarray) -> ndarray:
    '''
    Computationally more expensive compared to the Relu.
    '''
    return 1 / (1 + np.exp(-x))


# %% colab={"base_uri": "https://localhost:8080/", "height": 500} id="R6o9WX2BOK94" outputId="7b02d118-929c-48ac-9671-86076bce4c8e"
# https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html
fig, ax = plt.subplots(1, 3, sharey=True, figsize=(15, 5))

inputRange = np.arange(-2, 2, 0.01)
ax[0].plot(inputRange, square(inputRange))
ax[0].plot(inputRange, square(inputRange))
ax[0].set_title('Square function')
ax[0].set_xlabel('input')
ax[0].set_ylabel('output')

ax[1].plot(inputRange, leakyRelu(inputRange))
ax[1].plot(inputRange, leakyRelu(inputRange))
ax[1].set_title('"Leaky" Relu function')
ax[1].set_xlabel('input')
ax[1].set_ylabel('output')

ax[2].plot(inputRange, sigmoid(inputRange))
ax[2].plot(inputRange, sigmoid(inputRange))
ax[2].set_title('Sigmoid function')
ax[2].set_xlabel('input')
ax[2].set_ylabel('output')

# %% [markdown] id="arzeC4mIcZQ_"
# ## Simple deriv


# %% id="grdgQJffOK95"
def deriv(func: Callable[[ndarray], ndarray],
          input_: ndarray,
          delta: float = 0.001) -> ndarray:
    return (func(input_ + delta) - func(input_ - delta)) / (2 * delta)


# %% id="3y7aCotbOK96"
# A function takes in an ndarray as an argument and produces an ndarray
Array_Function = Callable[[ndarray], ndarray]

# A chain is a list of functions
Chain = List[Array_Function]

# %% [markdown] id="wxSiBm6FOK94"
# # Derivatives and Nested Functions

# %% [markdown] id="hXPgJX5zbyPz"
# ## Length = 2


# %% id="oHm2pX-lOK96"
def chain_length_2(chain: Chain, x: ndarray) -> ndarray:
    '''
    Evaluates two functions in a row, in a "chain"
    '''
    assert len(chain) == 2

    f1 = chain[0]
    f2 = chain[1]

    return f2(f1(x))


def chain_deriv_2(chain: Chain, input_range: ndarray) -> ndarray:
    '''
    Uses the chain rule to compute the derivative of two nested functions
    (f2(f1(x))' = f2'(f1(x)) * f1'(x)
    '''
    assert len(chain) == 2

    assert input_range.ndim == 1

    f1 = chain[0]
    f2 = chain[1]

    # df1/dx
    f1_of_x = f1(input_range)

    # df1/du
    df1dx = deriv(f1, input_range)

    # df2/du(f1(x))
    df2du = deriv(f2, f1(input_range))

    # Multiplying these quantities together at each point
    return df1dx * df2du


# %% id="8zgj-9qgOK97"
def plot_chain(ax,
               chain: Chain,
               input_range: ndarray,
               length: int = 2) -> None:
    '''
    Plots the designated chain function - a function made up of multiple 
    consecutive ndarray -> ndarray mappings - Across the input_range
    '''

    assert input_range.ndim == 1, \
    "Function requires a 1 dimensional ndarray as input_range"

    if length == 2:
        output_range = chain_length_2(chain, input_range)
    elif length == 3:
        output_range = chain_length_3(chain, input_range)
    ax.plot(input_range, output_range)


# %% id="EIbC9h-MOK98"
def plot_chain_deriv(ax,
                     chain: Chain,
                     input_range: ndarray,
                     length: int = 2) -> ndarray:
    '''
    Just plots the derivative of a nested function, aka the chain rule
    '''
    if length == 2:
        output_range = chain_deriv_2(chain, input_range)
    elif length == 3:
        output_range = chain_deriv_3(chain, input_range)
    ax.plot(input_range, output_range)


# %% colab={"base_uri": "https://localhost:8080/", "height": 751} id="Cp8N3fUbOK98" outputId="c253cb1a-0296-48c6-e101-3b8b00e09ad4"
# plot results of the chain rule
fig, ax = plt.subplots(1, 2, sharey=True, figsize=(16, 8))

PLOT_RANGE = np.arange(-3, 3, 0.01)

chain_1 = [square, sigmoid]
chain_2 = [sigmoid, square]

plot_chain(ax[0], chain_1, PLOT_RANGE)
plot_chain_deriv(ax[0], chain_1, PLOT_RANGE)

ax[0].legend(["$f(x)$", "$\\frac{df}{dx}$"])
ax[0].set_title("Function and derivative for \n$f(x) = sigmoid(square(x))$")

plot_chain(ax[1], chain_2, PLOT_RANGE)
plot_chain_deriv(ax[1], chain_2, PLOT_RANGE)

ax[1].legend(["$f(x)$", "$\\frac{df}{dx}$"])
ax[1].set_title("Function and derivative for \n$f(x) = square(sigmoid(x))$")

# %% [markdown] id="cwzbe-64bu5o"
# ## Length = 3


# %% id="vnHTqNfmOK99"
def chain_length_3(chain: Chain, x: ndarray) -> ndarray:
    '''
    Evaluates three functions in a row, in a "Chain" 
    This object is just used to convey how a chain of functions 
    is evaulated.
    '''
    assert len(chain) == 3

    f1 = chain[0]
    f2 = chain[1]
    f3 = chain[2]

    return f3(f2(f1(x)))


def chain_deriv_3(chain: Chain, input_range: ndarray) -> ndarray:
    '''
    This will calculate the derivative of three nested functions using the chain rule.
    f3(f2(f1(x)))' = f3'(f2(f1(x))) * f2'(f1(x)) * f1(x)'
    '''
    assert len(chain) == 3, \
    "This function requires 'Chain' objects to have length of 3"

    assert input_range.ndim == 1, \
    "Function requires a 1 dimensional ndarray as input_range"

    f1 = chain[0]
    f2 = chain[1]
    f3 = chain[2]

    # f1(x)
    f1_of_x = f1(input_range)

    # f2(f1(x))
    f2_of_x = f2(f1(input_range))

    # df3du
    df3du = deriv(f3, f2_of_x)

    # df2du
    df2du = deriv(f2, f1_of_x)

    # df1du
    df1du = deriv(f1, input_range)

    # Mutliply together at same point
    return df1du * df2du * df3du


# %% colab={"base_uri": "https://localhost:8080/", "height": 751} id="cIBc1IADOK99" outputId="e2a3d2e8-efc5-4ca8-87f4-13ed30b71187"
fig, ax = plt.subplots(1, 2, sharey=True, figsize=(16, 8))

chain_1 = [leakyRelu, square, sigmoid]
chain_2 = [leakyRelu, sigmoid, square]

PLOT_RANGE = np.arange(-3, 3, 0.01)
plot_chain(ax[0], chain_1, PLOT_RANGE, length=3)
plot_chain_deriv(ax[0], chain_1, PLOT_RANGE, length=3)

ax[0].legend(["$f(x)$", "$\\frac{df}{dx}$"])
ax[0].set_title(
    "Function and derivative for \n$f(x) = sigmoid(square(leakyRelu(x)))$")

plot_chain(ax[1], chain_2, PLOT_RANGE, length=3)
plot_chain_deriv(ax[1], chain_2, PLOT_RANGE, length=3)

ax[1].legend(["$f(x)$", "$\\frac{df}{dx}$"])
ax[1].set_title(
    "Function and derivative for \n$f(x) = square(sigmoid(leakyRelu(x)))$")

# %% [markdown] id="sZG_FeAjOK99"
# # Functions with Multiple Inputs


# %% id="D9XEReMuOK9-"
def multiple_inputs_add(x: ndarray, y: ndarray,
                        sigma: Array_Function) -> float:
    '''
    Function with multiple inputs, which will be added together
    Forward pass
    '''
    assert x.shape == y.shape

    a = x + y
    return sigma(a)


def multiple_inputs_add_backwards(x: ndarray, y: ndarray,
                                  sigma: Array_Function) -> float:
    '''
    Computes the derivative of two functions with respect to both inputs
    '''
    # compute the forward pass
    a = x + y

    # compute the backward pass
    dsda = deriv(sigma, a)

    # derivatives a with respect to each variable
    dadx, dady = 1, 1

    return dsda * dadx, dsda * dady


def multiple_inputs_multiply_backwards(x: ndarray, y: ndarray,
                                       sigma: Array_Function) -> float:
    '''
    Computes the derivative of two functions with respect to both inputs
    '''
    # compute the foward pass
    a = x * y

    # compute the backward pass
    dsda = deriv(sigma, a)

    # derivatives of a with respect to each constituent variable
    dadx, dady = y, x

    return dsda * dadx, dsda * dady


# %% [markdown] id="pbpEavrJOK9-"
# # Functions with Matrix (Multiple Vector) Inputs

# %% [markdown] id="UgRCAh0Ln-R1"
# ## X * W


# %% id="OAfEbHeAOK9-"
def matmul_forward(X: ndarray, W: ndarray) -> ndarray:
    '''
    Computes the forward pass of a matrix mulitplication
    '''
    assert X.shape[1] == W.shape[0], \
    '''
    For a dot product of X and W, 
    X must be a 1xn, not 1x{0} and 
    W must be a nx1, not {1}x1 
    which will give us a 1x1 matrix
    '''.format(X.shape[1], W.shape[0])

    # matrix multiplication
    N = np.dot(X, W)

    return N


# %% id="uenUNPNbOK9-"
def matmul_backward(X: ndarray, W: ndarray) -> ndarray:
    '''
    Book p24: 
    v(X, W) = N
    for dv/dX, dv/dx_i=w_i
    '''

    # backward pass
    dNdX = np.transpose(W, (1, 0))

    return dNdX


# %% colab={"base_uri": "https://localhost:8080/"} id="IS6NW4i_OK9_" outputId="566d05e6-1597-467a-867f-ba08767943a4"
np.random.seed(190203)

X = np.random.randn(1, 3)
W = np.random.randn(3, 1)

print(X)
matmul_backward(X, W)

# %% [markdown] id="52XSUOIdoDr6"
# ## f(X * W)
#
# In this case, we assign f(a) to Sigmoid


# %% id="xmzHUM2wOK9_"
def matrix_forward_extra(X: ndarray, W: ndarray,
                         sigma: Array_Function) -> ndarray:
    '''
    Computes the forward pass of a function involving mat mul
    with an additional function added to the end
    '''
    assert X.shape[1] == W.shape[0]

    # mat mul
    N = np.dot(X, W)

    # feed N through the additional function, sigma()
    S = sigma(N)

    return S


# %% id="ef6sJpY0OK9_"
def matrix_function_backward_extra(X: ndarray, W: ndarray,
                                   sigma: Array_Function) -> ndarray:
    '''
    Computes the derivative of our matrix function with
    respect to the first element
    '''
    assert X.shape[1] == W.shape[0]

    # matmul
    N = np.dot(X, W)

    # feeding the output of the matrix multiplication through sigma
    S = sigma(N)

    # backward calculation
    dSdN = deriv(sigma, N)

    # dNdX
    # dNdX = np.transpose(W, (1, 0))
    dNdX = matmul_backward(X, W)

    # multiply them together; dNdX is 1x1 so the order of the
    # operation will not matter
    return np.dot(dSdN, dNdX)


# %% colab={"base_uri": "https://localhost:8080/"} id="x0usrtZIOK-A" outputId="927e1a06-43fe-4efb-c7ab-fa1225d63342"
print(matrix_function_backward_extra(X, W, sigmoid))


# %% id="suVO6lFxOK-A"
def forward_test(ind1, ind2, inc):
    X1 = X.copy()
    X1[ind1, ind2] = X[ind1, ind2] + inc

    return matrix_forward_extra(X1, W, sigmoid)


# %% colab={"base_uri": "https://localhost:8080/"} id="3Mfrlk6UOK-A" outputId="e849bc47-8ee1-4913-bb45-51209c30a771"
(np.round(forward_test(0, 2, 0.01) - forward_test(0, 2, 0), 4)) / 0.01

# %% colab={"base_uri": "https://localhost:8080/"} id="ue-fFtLVOK-A" outputId="bcab697b-b926-459d-d419-9bdab0a449b7"
np.round(matrix_function_backward_extra(X, W, sigmoid)[0, 2], 2)

# %% [markdown] id="02d8vmvFOK-B"
# So as you can see these two are the same which means the gradients we have calculated are correct.

# %% [markdown] id="9iiB6lLoOK-B"
# ## sum(f(X * W))
#
# Notice that dsum/dS = 1


# %% id="EDekYqZ_OK-B"
def matrix_function_forward_sum(X: ndarray, W: ndarray,
                                sigma: Array_Function) -> float:
    '''
    Computing the result of the forward pass of the function
    with input ndarray X and W and the final function sigma
    '''
    assert X.shape[1] == W.shape[0]

    # matmul
    N = np.dot(X, W)

    # feed N through sigma
    S = sigma(N)

    # sum all elements
    L = np.sum(S)

    return L


# %% id="Fg_1ZvekOK-B"
def matrix_function_backward_sum_1(X: ndarray, W: ndarray,
                                   sigma: Array_Function) -> ndarray:
    '''
    Compute derivative of matrix function with a sum with respect to the
    first matrix input X
    '''
    assert X.shape[1] == W.shape[0]

    # matmul
    N = np.dot(X, W)

    # feeding the output matrix through sigma
    S = sigma(N)

    # sum elementwise
    L = np.sum(S)

    # derivatives will be referred to as their function names

    # dLdS - just 1's since L is a summation
    dLdS = np.ones_like(S)

    # dSdN
    dSdN = deriv(sigma, N)

    # dLdN
    dLdN = dLdS * dSdN

    # dNdX
    dNdX = np.transpose(W, (1, 0))

    dLdX = np.dot(dSdN, dNdX)

    return dLdX


# %% colab={"base_uri": "https://localhost:8080/"} id="8ceGMxSJOK-C" outputId="ce66c551-e493-4ac0-d47b-efea928db095"
np.random.seed(190204)
X = np.random.randn(3, 3)
W = np.random.randn(3, 2)

print("X: ")
print(X)
print()

print("L: ")
print(round(matrix_function_forward_sum(X, W, sigmoid), 4))
print()

print("dLdX: ")
print(matrix_function_backward_sum_1(X, W, sigmoid))

# %% [markdown] id="zES0H54_OK-D"
# ## Change x11 while holding X and W Constant

# %% colab={"base_uri": "https://localhost:8080/"} id="NhueAAYlOK-C" outputId="0297e1ff-d9da-402e-e38d-47693a1f908d"
# X1 = X.copy()  # np.copy is a shallow copy
X1 = copy.deepcopy(X)
X1[0, 0] += 0.001

# forward
dLdX_increase = round((
                    matrix_function_forward_sum(X1, W, sigmoid) - \
                    matrix_function_forward_sum(X, W, sigmoid)) / 0.001, 4)
print(dLdX_increase)

# backward
dLdX_increase_examine = matrix_function_backward_sum_1(X, W, sigmoid)

assert dLdX_increase == round(dLdX_increase_examine[0, 0], 4)


# %% id="WmpayVo2OK-D"
def matrix_function_forward_sum(X: ndarray,
                                W: ndarray,
                                sigma: Array_Function,
                                modify_x11: bool = False,
                                x11: float = 0.05) -> float:
    '''
    Computing the result of the forward pass of this function
    with input tensors X and W and function sigma
    '''
    assert X.shape[1] == W.shape[0]

    if modify_x11:
        X1 = X.copy()
        X1[0][0] = x11

    # matmul
    N = np.dot(X1, W)

    # feeding N through sigma
    S = sigma(N)

    # sum all elements
    L = np.sum(S)

    return L


# %% colab={"base_uri": "https://localhost:8080/"} id="cp9SOYrOOK-D" outputId="e753dd15-9dc7-41b0-9a57-b01949dc99cb"
print("X: ")
print(X)

# %% id="D4ovDsUnOK-D"
x11s = np.arange(X[0][0] - 1, X[0][0] + 1, 0.01)
Ls = [
    matrix_function_forward_sum(X, W, sigmoid, modify_x11=True, x11=x11)
    for x11 in x11s
]

# %% colab={"base_uri": "https://localhost:8080/", "height": 878} id="z1tEfa5_OK-D" outputId="b2abafcd-be25-49fe-ce31-f90742b51791"
plt.plot(x11s, Ls)
plt.title(
    "value of $L$ as $x_{11}$ changes while holding $X$ and $W$ constant")
plt.xlabel("$x_{11}$")
plt.ylabel("$L$")
