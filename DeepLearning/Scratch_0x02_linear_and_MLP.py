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

# %% [markdown] id="BYLDH0_4QYm_"
# # Top

# %% id="UmQpjDtl4KxS"
import numpy as np
from numpy import ndarray

import matplotlib.pyplot as plt
# %matplotlib inline

from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from typing import Callable, Dict, List, Tuple


# %% id="4vdu1sEh6iKt"
def set_default(figsize=(10, 10), dpi=100):
    plt.style.use(['dark_background', 'bmh'])
    plt.rc('axes', facecolor='k')
    plt.rc('figure', facecolor='k')
    plt.rc('figure', figsize=figsize, dpi=dpi)
    plt.rc('font', size=15)


set_default()

# %% id="VEqVhaMnQAPh"
TEST_ALL = True

# %% [markdown] id="GHaWAsgQ2u5e"
# # Dataset

# %% id="Pcxo98Sl4KxU" colab={"base_uri": "https://localhost:8080/"} outputId="af5fc504-acdc-43b4-870f-71a1fa1f2c4f"
boston = load_boston()
boston.keys()

# %% id="okk3tXa34KxU"
data = boston.data
target = boston.target
features = boston.feature_names

# %% colab={"base_uri": "https://localhost:8080/"} id="hkywTBEG8z_1" outputId="78d8fc8a-2511-4d61-b7f4-d47331b37fa6"
print(data.shape)
print(target.shape)
print(features)

# %% colab={"base_uri": "https://localhost:8080/"} id="ix3q4Jz29ZyT" outputId="5e9cb56b-97c6-47f2-f3ce-3a15e2ebafeb"
print(boston['DESCR'])

# %% id="9V8T3vJq4KxV"
# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
# Standardize features by removing the mean and scaling to unit variance
s = StandardScaler()
data = s.fit_transform(data)

# %% id="B43U-H3k4KxW"
X_train, X_test, y_train, y_test = train_test_split(data,
                                                    target,
                                                    test_size=0.3,
                                                    random_state=80718)
y_train, y_test = y_train.reshape(-1, 1), y_test.reshape(-1, 1)

# %% [markdown] id="xsOP8j7J4KxZ"
# # Model Error Function


# %% id="jTlWJfqK4Kxa"
def mae(preds: ndarray, actuals: ndarray):
    '''
    Computes mean absolute error
    '''
    return np.mean(np.abs(preds - actuals))


def rmse(preds: ndarray, actuals: ndarray):
    '''
    Computes root mean squared error
    '''
    return np.sqrt(np.mean(np.power(preds - actuals, 2)))


# %% [markdown] id="2MDDts2l4KxX"
# # sk-learn Linear Regression

# %% id="k30VWHqy4KxX"
lr = LinearRegression(fit_intercept=True)
lr.fit(X_train, y_train)
preds = lr.predict(X_test)

# %% colab={"base_uri": "https://localhost:8080/", "height": 927} id="35x23rK54KxX" outputId="4834ce53-3ecf-4292-d80b-92b2a54d18c7"
plt.xlabel("Predicted value")
plt.ylabel("Actual value")
plt.title("Predicted vs. Actual values for\nLinear Regression model")
plt.xlim([0, 51])
plt.ylim([0, 51])
plt.scatter(preds, y_test)
plt.plot([0, 51], [0, 51])

# %% [markdown] id="QPawIJ2t4KxY"
# ## [!] Deterine the Most Important Feature
#
# p57: a larger coefficient means that the feature is more important
#
# The last one (-4.19, index 12) is the most important feature
#

# %% colab={"base_uri": "https://localhost:8080/"} id="lOkDbArA4Kxa" outputId="e8f5c682-4d9e-43c2-ccf1-3b142d93e983"
np.round(lr.coef_, 2)

# %% colab={"base_uri": "https://localhost:8080/"} id="fnr-YGsNbpue" outputId="e15b5f19-02e8-455e-880e-cb3b7efb81ea"
# Preview the last feature
np.round(X_test[:, 12], 4)

# %% [markdown] id="CI62hpG14KxZ"
# As you can see this is a non linear relationship. As our "most important" feature increases our target decreases in a nonlinear manner.

# %% colab={"base_uri": "https://localhost:8080/", "height": 927} id="JlsES20C4KxZ" outputId="8f747d92-317a-4e33-c5a0-94bfba638255"
plt.scatter(X_test[:, 12], y_test)
plt.xlabel("Most important feature from our data")
plt.ylabel("Target")
plt.title("Relationship between most\nimportant feature and target")

# %% [markdown] id="p6IuWUy63v_Q"
# ## Model Eval

# %% colab={"base_uri": "https://localhost:8080/"} id="IMLLMYvj4Kxa" outputId="398b0009-1b1c-4f99-a204-dbf32127a65f"
print("Mean absolute error: ", round(mae(preds, y_test), 4), "\n"
      "Root mean squared error: ", round(rmse(preds, y_test), 4))

# %% [markdown] id="495UXhT54Kxb"
# # Manual Linear Regression
#
# Using These Gradients to Train the Model
#
# 1. Select a batch of data.
# 2. Run the forward pass of the model.
# 3. Run the backward pass of the model using the info computed on the forward pass.
# 4. Use the gradients computed on the backward pass to update the weights.


# %% id="prB3C-l94Kxb"
# def forward_linear_regression(
#         X_batch: ndarray, 
#         y_batch: ndarray,
#         weights: Dict[str, ndarray]) -> Tuple[float, Dict[str, ndarray]]:
#     '''
#     Forward pass for the step-by-step Linear regression
#     '''
#     # asser batch sizes of X and y are equal
#     assert X_batch.shape[0] == y_batch.shape[0]

#     # assert that we can matmul X with all weights
#     assert X_batch.shape[1] == weights['W'].shape[0]

#     # assert that B is a scalar
#     assert weigths['B'].shape[0] == weights['B'].shape[1] == 1

#     # Compute the operations on the forward pass
#     N = np.dot(X_batch, weitghts['W'])

#     P = N + weights['B']

#     loss = np.mean(np.power(y_batch - P, 2))

#     # save info computed on the forward pass
#     forward_info: Dict[str, ndarray] = {}
#     forward_info['X'] = X_batch
#     forward_info['N'] = N
#     forward_info['P'] = P
#     forward_info['y'] = y_batch

#     return loss, forward_info

# %% id="DIqepMNZ4Kxc"
# def to_2d_np(a: ndarray, type: str = "col") -> ndarray:
#     '''
#     Turns a 1D tensor into a 2D tensor
#     '''
#     assert a.ndim == 1, \
#     "Input tensors must be 1 dimensional"

#     if type == "col":
#         return a.reshape(-1, 1)
#     elif type == "row":
#         return a.reshape(1, -1)

# %% [markdown] id="84WuGK4-HVjU"
# ## Loss and Gradients

# %% id="AqMXNtdI4Kxc"
def loss_gradients(forward_info: Dict[str, ndarray],
                   weights: Dict[str, ndarray]) -> Dict[str, ndarray]:
    '''
    Compute dLdW and dLdB for the step-by-step linear reg model

    X \
        func_1 -> func_2 -> sum -> L
    W /        N    |    P
                    B 

    '''
    batch_size = forward_info['X'].shape[0]

    dLdP = -2 * (forward_info['y'] - forward_info['P'])

    dPdN = np.ones_like(forward_info['N'])

    dPdB = np.ones_like(weights['B'])

    # dLdN = dLdP * dPdN

    dNdW = np.transpose(forward_info['X'], (1, 0))

    # L, P, N, W
    # use matmul, dNdW must be on the left for the dimensions to align
    dLdW = np.dot(dNdW, dLdP * dPdN)

    # L, P, B
    # must sum along the dimension representing the batch size
    dLdB = (dLdP * dPdB).sum(axis=0)

    loss_grad: Dict[str, ndarray] = {}
    loss_grad['W'] = dLdW
    loss_grad['B'] = dLdB

    return loss_grad


# %% id="FoGn-TBM4Kxd"
def forward_loss(
        X: ndarray, 
        y: ndarray,
        weights: Dict[str, ndarray]) -> Tuple[Dict[str, ndarray], float]:
    '''
    Generate predictions and calculate loss
    '''
    N = np.dot(X, weights['W'])

    P = N + weights['B']  # prediction

    loss = np.mean(np.power(y - P, 2))  # mse

    forward_info: Dict[str, ndarray] = {}
    forward_info['X'] = X
    forward_info['N'] = N
    forward_info['P'] = P
    forward_info['y'] = y

    return forward_info, loss


# %% [markdown] id="cca513rd7jkf"
# ## Train Functions

# %% id="aVOrGv0w4Kxd"
def init_weights(n_in: int) -> Dict[str, ndarray]:
    '''
    Initialize weights on first forward pass of model
    '''
    weights: Dict[str, ndarray] = {}
    W = np.random.randn(n_in, 1)
    B = np.random.randn(1, 1)

    weights['W'] = W
    weights['B'] = B

    return weights


# %% id="U9v5pGPc4Kxd"
Batch = Tuple[ndarray, ndarray]


# %% id="lMSdH4D46G9P"
def generate_batch(X: ndarray,
                   y: ndarray,
                   start: int = 0,
                   batch_size: int = 10) -> Batch:
    '''
    Generate batch from X and y, given a start position
    '''
    assert X.ndim == y.ndim == 2, \
    "X and y must be 2 dimensional"

    if start + batch_size > X.shape[0]:
        batch_size = X.shape[0] - start

    X_batch, y_batch = X[start:start + batch_size], y[start:start + batch_size]

    return X_batch, y_batch


# %% id="mmSGNBth5346"
def permute_data(X: ndarray, y: ndarray):
    '''
    Permute X and y, using the same permutation, along axis=0
    '''
    perm = np.random.permutation(X.shape[0])
    return X[perm], y[perm]


# %% [markdown] id="2NfQe7451gY4"
# ## Train

# %% id="wgIVDYQK4Kxd"
def train(X: ndarray,
          y: ndarray,
          n_iter: int = 1000,
          learning_rate: float = 0.01,
          batch_size: int = 100,
          return_losses: bool = False,
          return_weights: bool = False,
          seed: int = 1) -> None:
    '''
    Train the model for a certain number of epochs
    '''
    if seed:
        np.random.seed(seed)
    start = 0

    # Initialize weights
    weights = init_weights(X.shape[1])

    # Permute data
    X, y = permute_data(X, y)

    if return_losses:
        losses = []

    for i in range(n_iter):

        # Generate batch
        if start >= X.shape[0]:
            X, y = permute_data(X, y)
            start = 0

        X_batch, y_batch = generate_batch(X, y, start, batch_size)
        start += batch_size

        # Train net using generated batch
        forward_info, loss = forward_loss(X_batch, y_batch, weights)

        if return_losses:
            losses.append(loss)

        loss_grads = loss_gradients(forward_info, weights)
        for key in weights.keys():
            weights[key] -= learning_rate * loss_grads[key]

    if return_weights:
        return losses, weights

    return None


# %% id="SCdqpyxu4Kxe"
train_info = train(X_train,
                   y_train,
                   n_iter=1000,
                   learning_rate=0.001,
                   batch_size=23,
                   return_losses=True,
                   return_weights=True,
                   seed=80718)
losses = train_info[0]
weights = train_info[1]

# %% colab={"base_uri": "https://localhost:8080/", "height": 832} id="AfZeBPn44Kxe" outputId="2a78bcf1-6c30-4350-ad60-01202834d4e8"
plt.plot(list(range(1000)), losses)


# %% [markdown] id="AZG1MUCi1GBd"
# ## Predict

# %% id="zNmLvk2D4Kxe"
def predict(X: ndarray, weights: Dict[str, ndarray]):
    '''
    Generate predictions from the manual linear regression
    '''
    N = np.dot(X, weights['W'])

    return N + weights['B']


# %% id="D3xD4CBS4Kxe"
preds = predict(X_test, weights)

# %% colab={"base_uri": "https://localhost:8080/"} id="FOWrlUBh4Kxe" outputId="95140738-eb8b-48fa-bd95-3575a8c792d8"
print("Mean absolute error: ", round(mae(preds, y_test), 4), "\n"
      "Root mean squared error: ", round(rmse(preds, y_test), 4))

# %% colab={"base_uri": "https://localhost:8080/"} id="WWp86PhO4Kxf" outputId="6c15ba2e-5d23-4588-e918-b8ff9d421cbf"
np.round(y_test.mean(), 4)

# %% colab={"base_uri": "https://localhost:8080/"} id="QvnaFor14Kxf" outputId="89e2dc8f-72b2-40e1-e89f-ecceb7d3a254"
np.round(rmse(preds, y_test) / y_test.mean(), 4)

# %% [markdown] id="BVyXgZWv4Kxf"
# the above metric shows that rmse is ~23% on average of y

# %% colab={"base_uri": "https://localhost:8080/", "height": 927} id="A4vi4rKl4Kxf" outputId="018f7959-e921-444b-98c8-d142c20ba893"
plt.xlabel("Predicted value")
plt.ylabel("Actual value")
plt.title("Predicted vs. Actual values for\ncustom linear regression model")
plt.xlim([0, 51])
plt.ylim([0, 51])
plt.scatter(preds, y_test)
plt.plot([0, 51], [0, 51])

# %% [markdown] id="csVxLb0R4Kxg"
# ## Compare Coefficients (sk-learn vs Manual)

# %% colab={"base_uri": "https://localhost:8080/"} id="_uisp7My4Kxg" outputId="53e51763-0f8e-4f96-f5c8-c204f2667d40"
np.round(weights['W'].reshape(-1), 4)

# %% colab={"base_uri": "https://localhost:8080/"} id="TqX7PskE4Kxg" outputId="7ba5d324-8a75-4921-840d-229b17057190"
np.round(lr.coef_, 4)

# %% colab={"base_uri": "https://localhost:8080/"} id="eEtonx9g4Kxg" outputId="ae838448-7aa1-467c-8f26-bdb3b1b2b116"
np.round(weights['B'], 4)

# %% colab={"base_uri": "https://localhost:8080/"} id="QwzoNfnn4Kxg" outputId="4306a418-eceb-48f3-d21e-991776a09619"
np.round(lr.intercept_, 4)

# %% [markdown] id="dPkyfGda4Kxh"
# The coefficients are the 'same' as in scikit learn and the manual linear regression, although you can see slight discrepancies in the weights, but both have very similiar intercepts. As shown earlier they had almost identical RMSE and MAE, but since the weights are different they are different lines. Similar errors, but 'different' lines can be attributed to the dataset, at least that is what I would conclude.

# %% [markdown] id="eElpNw6K4Kxh"
# ### [!] Theoretical relationship between most important feature (NO.12) and the target

# %% id="CYPBE3KR4Kxh"
NUM = 40
a = np.repeat(X_test[:, :-1].mean(axis=0, keepdims=True), NUM, axis=0)
b = np.linspace(-1.5, 3.5, NUM).reshape(NUM, 1)

# %% id="PQCTFJE54Kxh"
test_feature = np.concatenate([a, b], axis=1)
preds = predict(test_feature, weights)[:, 0]

# %% colab={"base_uri": "https://localhost:8080/", "height": 832} id="DZDqwXyB4Kxh" outputId="f79854b7-6b8d-4264-eb31-6ebb53c40684"
plt.scatter(np.array(test_feature[:, -1]), np.array(preds))
plt.ylim([6, 51])

# %% [markdown] id="d8EKUyjI4Kxh"
# # Manual Neural Network
#
# 1. A Bunch of Linear Regressions
# 2. A Nonlinear Function
# 3. Another Linear Regression


# %% [markdown] id="nvLDn2AdQ9vL"
# $$ \frac{\partial \sigma}{\partial u}(x) = \sigma(x) * (1 - \sigma(x)) $$ 

# %% id="-fBcTHMq4Kxi"
def sigmoid(x: ndarray) -> ndarray:
    return 1 / (1 + np.exp(-1.0 * x))


# %% colab={"base_uri": "https://localhost:8080/", "height": 893} id="eq8IDPkC4Kxi" outputId="d254cdf1-cbcb-44c9-cacf-2ea3fc2a238b"
plt.plot(np.arange(-5, 5, 0.01), sigmoid(np.arange(-5, 5, 0.01)))
plt.title("Sigmoid function plotted from x=-5 to x=5")
plt.xlabel("X")
plt.ylabel("$sigmoid(x)$")


# %% id="O7AXhyuu4Kxi"
def init_weights(input_size: int, hidden_size: int) -> Dict[str, ndarray]:
    '''
    Initialize weights during the forward pass for step by step neural network model.
    '''
    weights_nn: Dict[str, ndarray] = {}
    weights_nn['W1'] = np.random.randn(input_size, hidden_size)
    weights_nn['B1'] = np.random.randn(1, hidden_size)
    weights_nn['W2'] = np.random.randn(hidden_size, 1)
    weights_nn['B2'] = np.random.randn(1, 1)
    return weights_nn


# %% [markdown] id="igh3TWjSNI4O"
# ## Loss and Gradients

# %% id="bePeix5u4Kxj"
def loss_gradients(forward_info_nn: Dict[str, ndarray],
                   weights_nn: Dict[str, ndarray]) -> Dict[str, ndarray]:
    '''
    Compute the partial derivatives of the loss w/ respect to each 
    parameter in the neural net
    - dLdW2
    - dLdB2
    - dLdW1
    - dLdB1

    X \
        func_1 -> func_2 -> sigmoid -> func_3 -> func_4 -> loss_func -> L
    W1/        M1   |    N1              |    M2   |     P     |
                    B1                   W2        B2       real_Y
    '''
    dLdP = -(forward_info_nn['y'] - forward_info_nn['P'])

    dPdM2 = np.ones_like(forward_info_nn['M2'])

    dLdM2 = dLdP * dPdM2

    dPdB2 = np.ones_like(weights_nn['B2'])

    # target
    dLdB2 = (dLdP * dPdB2).sum(axis=0)

    dM2dW2 = np.transpose(forward_info_nn['O1'], (1, 0))

    # target
    dLdW2 = np.dot(dM2dW2, dLdP)

    dM2dO1 = np.transpose(weights_nn['W2'], (1, 0))

    dLdO1 = np.dot(dLdM2, dM2dO1)

    dO1dN1 = sigmoid(
        forward_info_nn['N1']) * (1 - sigmoid(forward_info_nn['N1']))

    dLdN1 = dLdO1 * dO1dN1

    # target
    dN1dB1 = np.ones_like(weights_nn['B1'])

    dN1dM1 = np.ones_like(forward_info_nn['M1'])

    dLdB1 = (dLdN1 * dN1dB1).sum(axis=0)

    dLdM1 = dLdN1 * dN1dM1

    dM1dW1 = np.transpose(forward_info_nn['X'], (1, 0))

    # target
    dLdW1 = np.dot(dM1dW1, dLdM1)

    loss_gradients_nn: Dict[str, ndarray] = {}
    loss_gradients_nn['W2'] = dLdW2
    loss_gradients_nn['B2'] = dLdB2.sum(axis=0)
    loss_gradients_nn['W1'] = dLdW1
    loss_gradients_nn['B1'] = dLdB1.sum(axis=0)

    return loss_gradients_nn


# %% id="mpCX2qPF4Kxi"
def forward_loss_nn(
        X: ndarray, y: ndarray,
        weights_nn: Dict[str, ndarray]) -> Tuple[Dict[str, ndarray], float]:
    '''
    COmpute the forward pass and the loss for the manaul nerual net
    '''
    M1 = np.dot(X, weights_nn['W1'])

    N1 = M1 + weights_nn['B1']

    O1 = sigmoid(N1)

    M2 = np.dot(O1, weights_nn['W2'])

    P = M2 + weights_nn['B2']

    loss_nn = np.mean(np.power(y - P, 2))  # mse

    forward_info_nn: Dict[str, ndarray] = {}
    forward_info_nn['X'] = X
    forward_info_nn['M1'] = M1
    forward_info_nn['N1'] = N1
    forward_info_nn['O1'] = O1
    forward_info_nn['M2'] = M2
    forward_info_nn['P'] = P
    forward_info_nn['y'] = y

    return forward_info_nn, loss_nn


# %% [markdown] id="1AUmV4k9PgM9"
# ## Train

# %% id="MgbV_KKY4Kxk"
def train(X_train: ndarray,
          y_train: ndarray,
          X_test: ndarray,
          y_test: ndarray,
          n_iter: int = 1000,
          test_every: int = 1000,
          learning_rate: float = 0.01,
          hidden_size: int = 13,
          batch_size: int = 100,
          return_losses: bool = False,
          return_weights: bool = False,
          return_scores: bool = False,
          seed: int = 1) -> None:

    if seed:
        np.random.seed(seed)

    start = 0

    # Intitialize weights
    weights_nn = init_weights(X_train.shape[1], hidden_size=hidden_size)

    # Permute data
    X_train, y_train = permute_data(X_train, y_train)

    losses = []

    val_scores = []

    for i in range(n_iter):

        # Generate batch
        if start >= X_train.shape[0]:
            X_train, y_train = permute_data(X_train, y_train)
            start = 0

        X_batch, y_batch = generate_batch(X_train, y_train, start, batch_size)
        start += batch_size

        # train net using generated batch
        forward_info_nn, loss = forward_loss_nn(X_batch, y_batch, weights_nn)

        if return_losses:
            losses.append(loss)

        loss_grads_nn = loss_gradients(forward_info_nn, weights_nn)
        for key in weights_nn.keys():
            weights_nn[key] -= learning_rate * loss_grads_nn[key]

        if return_scores:
            if i % test_every == 0 and i != 0:
                preds = predict(X_test, weights_nn)
                val_scores.append(r2_score(preds, y_test))
    if return_weights:
        return losses, weights_nn, val_scores

    return None


# %% [markdown] id="nQGZMrTuPZzQ"
# ## Predict

# %% id="ZlbR7ysJ4Kxj"
def predict(X: ndarray, weights_nn: Dict[str, ndarray]) -> ndarray:
    '''
    Generate predictions from the manual neural net model
    '''
    M1 = np.dot(X, weights_nn['W1'])

    N1 = M1 + weights_nn['B1']

    O1 = sigmoid(N1)

    M2 = np.dot(O1, weights_nn['W2'])

    P = M2 + weights_nn['B2']

    return P


# %% id="z-15miZf4Kxk"
if TEST_ALL:
    num_iter = 10000
    test_every = 1000
    train_info = train(X_train, 
                       y_train, 
                       X_test, 
                       y_test,
                       n_iter=num_iter,
                       test_every = test_every,
                       learning_rate = 0.001,
                       batch_size=23, 
                       return_losses=True, 
                       return_weights=True, 
                       return_scores=True,
                       seed=80718)
    losses = train_info[0]
    weights = train_info[1]
    val_scores = train_info[2]

# %% colab={"base_uri": "https://localhost:8080/"} id="zjtxSvwkRpWv" outputId="59720e4b-1e61-4394-cc8b-2639a36156a7"
val_scores

# %% id="GFJdW1qR4Kxk" colab={"base_uri": "https://localhost:8080/", "height": 876} outputId="ccd73634-aa92-4517-b45c-e44b3b8437af"
if TEST_ALL:
    plt.ylim([-1, 1])
    plt.plot(list(range(int(num_iter / test_every - 1))), val_scores); 
    plt.xlabel("Batches (000s)")
    plt.title("Validation Scores")

# %% [markdown] id="UFIMPSwu4Kxk"
# ## Learning rate tuning


# %% id="ujQIfrdQ4Kxl"
def r2_score_lr(learning_rate):
    train_info_nn = train(X_train,
                          y_train,
                          X_test,
                          y_test,
                          n_iter=100000,
                          test_every=100000,
                          learning_rate=learning_rate,
                          batch_size=23,
                          return_losses=True,
                          return_weights=True,
                          return_scores=True,
                          seed=80718)
    weights_nn = train_info_nn[1]
    preds = predict(X_test, weights_nn)
    return r2_score(y_test, preds)


# %% id="m6WBVft24Kxl"
if TEST_ALL:
    lrs = np.geomspace(1e-2, 1e-6, num=20)

# %% id="ZBHfvI7U4Kxl"
if TEST_ALL:
    r2s = [r2_score_lr(lr) for lr in lrs]

# %% id="XWnLNFuG4Kxl" colab={"base_uri": "https://localhost:8080/", "height": 822} outputId="ea8489d2-5a17-42dd-bcf1-5e1afbc91c54"
if TEST_ALL:
    # Make a plot with log scaling on the x axis
    plt.semilogx(lrs, r2s)

# %% colab={"base_uri": "https://localhost:8080/"} id="ofyseZ_lWMeA" outputId="67e64eab-1a4f-439c-fe92-d62ca8547115"
min_idx = r2s.index(min(r2s))
print('Best lr: {}, Minimum loss: {}'.format(lrs[min_idx], min(r2s)))

# %% [markdown] id="O4g_Zck04Kxl"
# ## Evaluating best model

# %% id="f3yVHewK4Kxl"
train_info_nn = train(X_train,
                      y_train,
                      X_test,
                      y_test,
                      n_iter=4000,
                      test_every=1000,
                      learning_rate=0.001,
                      batch_size=23,
                      return_losses=True,
                      return_weights=True,
                      return_scores=True,
                      seed=180807)
losses = train_info_nn[0]
weights_nn = train_info_nn[1]

# %% colab={"base_uri": "https://localhost:8080/", "height": 832} id="o-P94eC54Kxm" outputId="be9706c3-5dba-44f3-d502-204dd7921694"
plt.plot(list(range(4000)), losses)

# %% id="4xehoihw4Kxm"
preds = predict(X_test, weights_nn)

# %% [markdown] id="jocmYnbw4Kxm"
# ##  Investigation of most important features
# Most important combination of features are the two with abs() > 9:

# %% colab={"base_uri": "https://localhost:8080/"} id="Dzx3sJ2f4Kxm" outputId="5f16f09f-71d9-49b5-dbd8-306b2ffbd2b8"
weights_nn['W2']

# %% [markdown] id="fDzCqCim4Kxm"
# The combinations that fit the criteria above as index 7 and 9

# %% colab={"base_uri": "https://localhost:8080/"} id="i1HLxyzu4Kxm" outputId="d6907bf2-01bb-496b-bac7-2160bf0535c3"
print("Weights example (7th): \n", weights_nn['W1'][7])

# %% colab={"base_uri": "https://localhost:8080/"} id="i_cQQOG_4Kxn" outputId="226d9055-c724-4968-b960-f69c604e1fa9"
print("Mean abs error: ", round(mae(preds, y_test), 4), "\n"
      "Root mean sqr error: ", round(rmse(preds, y_test), 4))

# %% colab={"base_uri": "https://localhost:8080/", "height": 927} id="YcvlobYT4Kxn" outputId="49a09e85-3a87-42ff-be35-e1c9281b0761"
plt.xlabel("Predicted value")
plt.ylabel("Target")
plt.title("Predicted value vs. target, \n neural network regression")
plt.xlim([0, 51])
plt.ylim([0, 51])
plt.scatter(preds, y_test)
plt.plot([0, 51], [0, 51])

# %% colab={"base_uri": "https://localhost:8080/"} id="71MySuA34Kxn" outputId="b66304f7-044b-404a-aed3-4152bf4d400a"
np.round(np.mean(np.array(np.abs(preds - y_test))), 4)

# %% colab={"base_uri": "https://localhost:8080/"} id="WVkhC5f94Kxn" outputId="220e040c-0bc0-4b5b-94d2-b8c21f699437"
np.round(np.mean(np.array(np.power(preds - y_test, 2))), 4)

# %% [markdown] id="kf7Pr5yU4Kxn"
# ### [!] Theoretical relationship between most important feature (NO.12) and target

# %% id="2tFphXYc4Kxo"
NUM = 40
a = np.repeat(X_test[:, :-1].mean(axis=0, keepdims=True), NUM, axis=0)
b = np.linspace(-1.5, 3.5, NUM).reshape(NUM, 1)
test_feature = np.concatenate([a, b], axis=1)
preds_test = predict(test_feature, weights_nn)[:, 0]

# %% colab={"base_uri": "https://localhost:8080/", "height": 927} id="rXHRYiXw4Kxo" outputId="3897519a-de47-445e-9f78-851bad592672"
plt.scatter(X_test[:, 12], preds)
plt.plot(np.array(test_feature[:, -1]), preds_test, linewidth=2, c='orange')
plt.ylim([6, 51])
plt.xlabel("Most important feature (normalized)")
plt.ylabel("Target/Predictions")
plt.title(
    "Most important feature vs target and predictions \n Manual neural net regression"
)
