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

# %% id="nHRZC9-g0KUT"
import numpy as np
from numpy import ndarray

from typing import List


# %% id="LvMDboYG0KUX"
def assert_same_shape(array: ndarray, array_grad: ndarray):
    assert array.shape == array_grad.shape, \
        '''
        Two ndarrays should have the same shape;
        instead, first ndarray's shape is {0}
        and second ndarray's shape is {1}.
        '''.format(tuple(array_grad.shape), tuple(array.shape))
    return None


# %% [markdown] id="DBwiDZ370KUY"
# # `Operation` and `ParamOperation`

# %% [markdown] id="hOGraSB18Mta"
# ## Base class (No extra parameter)


# %% id="gsxfsV9a0KUZ"
class Operation(object):
    '''
    Base class for an "operation" in a neural network.
    '''
    def __init__(self):
        pass

    def forward(self, input_: ndarray):
        '''
        Stores input in the self._input instance variable
        Calls the self._output() function.
        '''
        self.input_ = input_

        self.output = self._output()

        return self.output

    def backward(self, output_grad: ndarray) -> ndarray:
        '''
        Calls the self._input_grad() function.
        Checks that the appropriate shapes match.
        '''
        assert_same_shape(self.output, output_grad)

        self.input_grad = self._input_grad(output_grad)

        assert_same_shape(self.input_, self.input_grad)
        return self.input_grad

    def _output(self) -> ndarray:
        '''
        The _output method must be defined for each Operation
        '''
        raise NotImplementedError()

    def _input_grad(self, output_grad: ndarray) -> ndarray:
        '''
        The _input_grad method must be defined for each Operation
        '''
        raise NotImplementedError()


# %% [markdown] id="Z2pGqT2DB9Kh"
# ## Base class (With extra parameter)
#
# - Need a `_param_grad`
# - May output different shape


# %% id="grkiOaRz0KUa"
class ParamOperation(Operation):
    '''
    An Operation with parameters.
    '''
    def __init__(self, param: ndarray) -> ndarray:
        '''
        The ParamOperation method
        '''
        super().__init__()
        self.param = param

    def backward(self, output_grad: ndarray) -> ndarray:
        '''
        Calls self._input_grad and self._param_grad.
        Checks appropriate shapes.
        '''

        assert_same_shape(self.output, output_grad)

        self.input_grad = self._input_grad(output_grad)
        self.param_grad = self._param_grad(output_grad)

        assert_same_shape(self.input_, self.input_grad)
        assert_same_shape(self.param, self.param_grad)

        return self.input_grad

    def _param_grad(self, output_grad: ndarray) -> ndarray:
        '''
        Every subclass of ParamOperation must implement _param_grad.
        '''
        raise NotImplementedError()


# %% [markdown] id="jMTmEGie0KUa"
# ## WeightMultiply
#
# - `self.input` is defined in Operation


# %% id="aB4CH94_0KUb"
class WeightMultiply(ParamOperation):
    '''
    Weight multiplication operation for a neural network.
    '''
    def __init__(self, W: ndarray):
        '''
        Initialize Operation with self.param = W.
        '''
        super().__init__(W)

    def _output(self) -> ndarray:
        '''
        Compute output.
        '''
        return np.dot(self.input_, self.param)

    def _input_grad(self, output_grad: ndarray) -> ndarray:
        '''
        Compute input gradient.
        '''
        return np.dot(output_grad, np.transpose(self.param, (1, 0)))

    def _param_grad(self, output_grad: ndarray) -> ndarray:
        '''
        Compute parameter gradient.
        '''
        return np.dot(np.transpose(self.input_, (1, 0)), output_grad)


# %% [markdown] id="T7IDjllA8S5u"
# ## BiasAdd


# %% id="SGXiEpcG0KUc"
class BiasAdd(ParamOperation):
    '''
    Compute bias addition.
    '''
    def __init__(self, B: ndarray):
        '''
        Initialize Operation with self.param = B.
        Check appropriate shape.
        '''
        assert B.shape[0] == 1

        super().__init__(B)

    def _output(self) -> ndarray:
        '''
        Compute output.
        '''
        return self.input_ + self.param

    def _input_grad(self, output_grad: ndarray) -> ndarray:
        '''
        Compute input gradient.
        '''
        return np.ones_like(self.input_) * output_grad

    def _param_grad(self, output_grad: ndarray) -> ndarray:
        '''
        Compute parameter gradient.
        '''
        param_grad = np.ones_like(self.param) * output_grad
        return np.sum(param_grad, axis=0).reshape(1, param_grad.shape[1])


# %% [markdown] id="Zib_Jsas8WOz"
# ## Sigmoid


# %% id="7S4gRoVk0KUc"
class Sigmoid(Operation):
    '''
    Sigmoid activation function.
    '''
    def __init__(self) -> None:
        '''Pass'''
        super().__init__()

    def _output(self) -> ndarray:
        '''
        Compute output.
        '''
        return 1.0 / (1.0 + np.exp(-1.0 * self.input_))

    def _input_grad(self, output_grad: ndarray) -> ndarray:
        '''
        Compute input gradient.
        '''
        sigmoid_backward = self.output * (1.0 - self.output)
        input_grad = sigmoid_backward * output_grad
        return input_grad


# %% [markdown] id="0LGGpYpK8ZnY"
# ## Linear


# %% id="PsV1UmFF0KUd"
class Linear(Operation):
    '''
    "Identity" activation function
    '''
    def __init__(self) -> None:
        '''Pass'''
        super().__init__()

    def _output(self) -> ndarray:
        '''Pass through'''
        return self.input_

    def _input_grad(self, output_grad: ndarray) -> ndarray:
        '''Pass through'''
        return output_grad


# %% [markdown] id="rnk14rMX0KUd"
# # `Layer` and `Dense`
#
# - Defining the correct series of Operations in the `_setup_layer` function
# - Initializing and storing the parameters in these Operations
# - The forward and backward methods simply involve sending the input successively forward through a series of Operations

# %% [markdown] id="myo7VMYpKCSi"
# ## Layer class


# %% id="ImETQht_0KUe"
class Layer(object):
    '''
    A "layer" of neurons in a neural network.
    '''
    def __init__(self, neurons: int):
        '''
        The number of "neurons" roughly corresponds to the "breadth" of the layer
        '''
        self.neurons = neurons
        self.first = True
        self.params: List[ndarray] = []
        self.param_grads: List[ndarray] = []
        self.operations: List[Operation] = []  # reversed in backward

    def _setup_layer(self, num_in: int) -> None:
        '''
        The _setup_layer function must be implemented for each layer
        '''
        raise NotImplementedError()

    def forward(self, input_: ndarray) -> ndarray:
        '''
        Passes input forward through a series of operations
        '''
        if self.first:
            self._setup_layer(input_)
            self.first = False

        self.input_ = input_

        for operation in self.operations:
            input_ = operation.forward(input_)

        self.output = input_

        return self.output

    def backward(self, output_grad: ndarray) -> ndarray:
        '''
        Passes output_grad backward through a series of operations
        Checks appropriate shapes
        '''

        assert_same_shape(self.output, output_grad)

        for operation in reversed(self.operations):
            output_grad = operation.backward(output_grad)

        input_grad = output_grad

        self._param_grads()

        return input_grad

    def _param_grads(self) -> ndarray:
        '''
        Extracts the _param_grads from a layer's operations
        '''

        self.param_grads = []
        for operation in self.operations:
            if issubclass(operation.__class__, ParamOperation):
                self.param_grads.append(operation.param_grad)

    def _params(self) -> ndarray:
        '''
        Extracts the _params from a layer's operations
        '''

        self.params = []
        for operation in self.operations:
            if issubclass(operation.__class__, ParamOperation):
                self.params.append(operation.param)


# %% [markdown] id="VB9ffl4tKEMd"
# ## Dense Layer
#
# - For Parameter Operations (e.g. BiasAdd etc.)
# - Make the default activation a Linear activation (No activation)


# %% id="Zpc0Pw9u0KUe"
class Dense(Layer):
    '''
    A fully connected layer which inherits from "Layer"
    '''
    def __init__(self, neurons: int, activation: Operation = Sigmoid()):
        '''
        Requires an activation function upon initialization
        '''
        super().__init__(neurons)
        self.activation = activation

    def _setup_layer(self, input_: ndarray) -> None:
        '''
        Defines the operations of a fully connected layer.
        '''
        if self.seed:
            np.random.seed(self.seed)

        self.params = []

        # weights
        self.params.append(np.random.randn(input_.shape[1], self.neurons))

        # bias
        self.params.append(np.random.randn(1, self.neurons))

        self.operations = [
            WeightMultiply(self.params[0]),
            BiasAdd(self.params[1]), self.activation
        ]

        return None


# %% [markdown] id="1MsKMYau0KUf"
# # `Loss`

# %% [markdown] id="-8KdmzVRLp4I"
# ## Loss class


# %% id="SEvUJPKR0KUf"
class Loss(object):
    '''
    The "loss" of a neural network
    '''
    def __init__(self):
        '''Pass'''
        pass

    def forward(self, prediction: ndarray, target: ndarray) -> float:
        '''
        Computes the actual loss value
        '''
        assert_same_shape(prediction, target)

        self.prediction = prediction
        self.target = target

        loss_value = self._output()

        return loss_value

    def backward(self) -> ndarray:
        '''
        Computes gradient of the loss value with respect to the input to the loss function
        '''
        self.input_grad = self._input_grad()

        assert_same_shape(self.prediction, self.input_grad)

        return self.input_grad

    def _output(self) -> float:
        '''
        Every subclass of "Loss" must implement the _output function.
        '''
        raise NotImplementedError()

    def _input_grad(self) -> ndarray:
        '''
        Every subclass of "Loss" must implement the _input_grad function.
        '''
        raise NotImplementedError()


# %% [markdown] id="0gGyW3LILkEH"
# ## MSE


# %% id="4eudcmzn0KUf"
class MeanSquaredError(Loss):
    def __init__(self) -> None:
        '''Pass'''
        super().__init__()

    def _output(self) -> float:
        '''
        Computes the per-observation squared error loss
        '''
        loss = (np.sum(np.power(self.prediction - self.target, 2)) /
                self.prediction.shape[0])

        return loss

    def _input_grad(self) -> ndarray:
        '''
        Computes the loss gradient with respect to the input for MSE loss
        '''

        return 2.0 * (self.prediction - self.target) / self.prediction.shape[0]


# %% [markdown] id="PdJ-NF2o0KUg"
# # `Neural Network`
#
# - Check p85
# - Have a list of Layers, each Layer will have a list of Operations (have `forward` and `backward`)
# - Batch Training
#
# TODO: Add `_update_params`
#


# %% id="KZAriXyS0KUg"
class NeuralNetwork(object):
    '''
    The class for a neural network.
    '''
    def __init__(self, layers: List[Layer], loss: Loss, seed: int = 1) -> None:
        '''
        Neural networks need layers, and a loss.
        '''
        self.layers = layers
        self.loss = loss
        self.seed = seed
        if seed:
            for layer in self.layers:
                setattr(layer, "seed", self.seed)

    def forward(self, x_batch: ndarray) -> ndarray:
        '''
        Passes data forward through a series of layers.
        '''
        x_out = x_batch
        for layer in self.layers:
            x_out = layer.forward(x_out)

        return x_out

    def backward(self, loss_grad: ndarray) -> None:
        '''
        Passes data backward through a series of layers.
        '''

        grad = loss_grad
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

        return None

    def train_batch(self, x_batch: ndarray, y_batch: ndarray) -> float:
        '''
        Passes data forward through the layers.
        Computes the loss.
        Passes data backward through the layers.
        '''

        predictions = self.forward(x_batch)

        loss = self.loss.forward(predictions, y_batch)

        self.backward(self.loss.backward())

        return loss

    def params(self):
        '''
        Gets the parameters for the network.
        '''
        for layer in self.layers:
            yield from layer.params

    def param_grads(self):
        '''
        Gets the gradient of the loss with respect to the parameters for the network.
        '''
        for layer in self.layers:
            yield from layer.param_grads


# %% [markdown] id="mrFpJqwO0KUg"
# # `Optimizer` and `SGD`

# %% [markdown] id="C963nTrZMdiL"
# ## Optimizer class


# %% id="mw-a6I0d0KUh"
class Optimizer(object):
    '''
    Base class for a neural network optimizer.
    '''
    def __init__(self, lr: float = 0.01):
        '''
        Every optimizer must have an initial learning rate.
        '''
        self.lr = lr

    def step(self) -> None:
        '''
        Every optimizer must implement the "step" function.
        '''
        pass


# %% [markdown] id="7wenS_lQMfUZ"
# ## SGD


# %% id="e2zlfPkH0KUh"
class SGD(Optimizer):
    '''
    Stochasitc gradient descent optimizer.
    '''
    def __init__(self, lr: float = 0.01) -> None:
        super().__init__(lr)

    def step(self):
        '''
        For each parameter, adjust in the appropriate direction, with the magnitude of the adjustment 
        based on the learning rate.
        '''
        for (param, param_grad) in zip(self.net.params(),
                                       self.net.param_grads()):

            param -= self.lr * param_grad


# %% [markdown] id="PWoH40rH0KUh"
# # `Trainer` class
#
# - Generate batches
# - Permute data for training
# - Early stopping
#

# %% id="s6-hq2Vx0KUh"
from copy import deepcopy
from typing import Tuple


class Trainer(object):
    '''
    Trains a neural network
    '''
    def __init__(self, net: NeuralNetwork, optim: Optimizer) -> None:
        '''
        Requires a neural network and an optimizer in order for training to occur. 
        Assign the neural network as an instance variable to the optimizer.
        '''
        self.net = net
        self.optim = optim
        self.best_loss = 1e9
        setattr(self.optim, 'net', self.net)

    def generate_batches(self,
                         X: ndarray,
                         y: ndarray,
                         size: int = 32) -> Tuple[ndarray]:
        '''
        Generates batches for training 
        '''
        assert X.shape[0] == y.shape[0], \
        '''
        features and target must have the same number of rows, instead
        features has {0} and target has {1}
        '''.format(X.shape[0], y.shape[0])

        N = X.shape[0]

        for ii in range(0, N, size):
            X_batch, y_batch = X[ii:ii + size], y[ii:ii + size]

            yield X_batch, y_batch

    def fit(self,
            X_train: ndarray,
            y_train: ndarray,
            X_test: ndarray,
            y_test: ndarray,
            epochs: int = 100,
            eval_every: int = 10,
            batch_size: int = 32,
            seed: int = 1,
            restart: bool = True) -> None:
        '''
        Fits the neural network on the training data for a certain number of epochs.
        Every "eval_every" epochs, it evaluated the neural network on the testing data.
        '''

        np.random.seed(seed)
        if restart:
            for layer in self.net.layers:
                layer.first = True

            self.best_loss = 1e9

        for e in range(epochs):
            if (e + 1) % eval_every == 0:
                # for early stopping
                last_model = deepcopy(self.net)

            X_train, y_train = permute_data(X_train, y_train)

            batch_generator = self.generate_batches(X_train, y_train,
                                                    batch_size)

            for ii, (X_batch, y_batch) in enumerate(batch_generator):
                self.net.train_batch(X_batch, y_batch)
                self.optim.step()

            if (e + 1) % eval_every == 0:
                test_preds = self.net.forward(X_test)
                loss = self.net.loss.forward(test_preds, y_test)

                if loss < self.best_loss:
                    print(f"Validation loss after {e+1} epochs is {loss:.3f}")
                    self.best_loss = loss
                else:
                    print(
                        f"""Loss increased after epoch {e+1}, final loss was {self.best_loss:.3f}, \
                        using the model from epoch {e+1-eval_every}"""
                    )
                    self.net = last_model
                    # ensure self.optim is still updating self.net
                    setattr(self.optim, 'net', self.net)
                    break


# %% [markdown] id="bQbDd6MD0KUi"
# # Evaluation metrics


# %% id="ovy5BOsd0KUi"
def mae(y_true: ndarray, y_pred: ndarray):
    '''
    Compute mean absolute error for a neural network.
    '''
    return np.mean(np.abs(y_true - y_pred))


def rmse(y_true: ndarray, y_pred: ndarray):
    '''
    Compute root mean squared error for a neural network.
    '''
    return np.sqrt(np.mean(np.power(y_true - y_pred, 2)))


def eval_regression_model(model: NeuralNetwork, X_test: ndarray,
                          y_test: ndarray):
    '''
    Compute mae and rmse for a neural network.
    '''
    preds = model.forward(X_test)
    preds = preds.reshape(-1, 1)
    print("Mean absolute error: {:.2f}".format(mae(preds, y_test)))
    print()
    print("Root mean squared error {:.2f}".format(rmse(preds, y_test)))


# %% [markdown] id="11xE2HmXOwMW"
# # Define Neural Networks

# %% id="ePEdjegl0KUj"
lr = NeuralNetwork(
    layers=[Dense(neurons=1, activation=Linear())],
    loss=MeanSquaredError(),
    seed=20190501)

nn = NeuralNetwork(
    layers=[
        Dense(neurons=13, activation=Sigmoid()),
        Dense(neurons=1, activation=Linear())
    ],
    loss=MeanSquaredError(),
    seed=20190501)

dl = NeuralNetwork(
    layers=[
        Dense(neurons=13, activation=Sigmoid()),
        Dense(neurons=13, activation=Sigmoid()),
        Dense(neurons=1, activation=Linear())
    ],
    loss=MeanSquaredError(),
    seed=20190501)

# %% [markdown] id="d9C2_sTO0KUj"
# # Read in the data, train-test split etc.

# %% id="PYITKScp0KUj"
from sklearn.datasets import load_boston

boston = load_boston()
data = boston.data
target = boston.target
features = boston.feature_names

# %% id="J33nz1Ef0KUj"
# Scaling the data
from sklearn.preprocessing import StandardScaler

s = StandardScaler()
data = s.fit_transform(data)


# %% id="StB04B7j0KUj"
def to_2d_np(a: ndarray, type: str = "col") -> ndarray:
    '''
    Turns a 1D Tensor into 2D
    '''

    assert a.ndim == 1, \
    "Input tensors must be 1 dimensional"

    if type == "col":
        return a.reshape(-1, 1)
    elif type == "row":
        return a.reshape(1, -1)


# %% id="EzdJw_Us0KUj"
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data,
                                                    target,
                                                    test_size=0.3,
                                                    random_state=80718)

# make target 2d array
y_train, y_test = to_2d_np(y_train), to_2d_np(y_test)

# %% [markdown] id="o7dgZopI0KUk"
# # Train the three models

# %% id="_m2MakvR0KUk"
# helper function


def permute_data(X, y):
    perm = np.random.permutation(X.shape[0])
    return X[perm], y[perm]


# %% id="B1vyyPto0KUk" outputId="89fe3b13-420f-4822-9240-cd9e36c6a80d"
trainer = Trainer(lr, SGD(lr=0.01))

trainer.fit(X_train,
            y_train,
            X_test,
            y_test,
            epochs=50,
            eval_every=10,
            seed=20190501)
print()
eval_regression_model(lr, X_test, y_test)

# %% id="cM5FmZk30KUl" outputId="355d53c0-70a8-4f12-ff7f-0ecc3fbafe9a"
trainer = Trainer(nn, SGD(lr=0.01))

trainer.fit(X_train,
            y_train,
            X_test,
            y_test,
            epochs=50,
            eval_every=10,
            seed=20190501)
print()
eval_regression_model(nn, X_test, y_test)

# %% id="bFeeLVNP0KUl" outputId="fdacc8ec-19d3-477b-eda0-493811db32ac"
trainer = Trainer(dl, SGD(lr=0.01))

trainer.fit(X_train,
            y_train,
            X_test,
            y_test,
            epochs=50,
            eval_every=10,
            seed=20190501)
print()
eval_regression_model(dl, X_test, y_test)
