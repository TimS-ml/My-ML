# ---
# jupyter:
#   jupytext:
#     comment_magics: false
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python (torch)
#     language: python
#     name: torch
# ---

# %% [markdown] colab_type="text" id="A--sgCKzn9oD"
# # Linear Regression

# %% [markdown] colab_type="text" id="KFBDBTV_n9oE"
# ## Simple input

# %% colab={} colab_type="code" id="v5Gd8ez6n9oF"
from torch import nn
import torch
from torch import tensor

# %% colab={} colab_type="code" id="giA9i3Unn9oM"
# Training data and ground truth
x_data = tensor([[1.0], [2.0], [3.0]])
y_data = tensor([[2.0], [4.0], [6.0]])


# %% colab={} colab_type="code" id="uEytgyTRn9oS"
class Model(nn.Module):
    def __init__(self):
        """
        In the constructor we instantiate two nn.Linear module
        """
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(1, 1)  # One in and one out

    def forward(self, x):
        """
        In the forward function we accept a Variable of input data and we must return
        a Variable of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Variables.
        """
        y_pred = self.linear(x)
        return y_pred


# %% colab={} colab_type="code" id="ZWMZ2Zl-n9oW" outputId="3e22e72b-4a95-465b-e87d-48410d2cdbba" tags=["outputPrepend"]
# our model
model = Model()

# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the model.
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Training loop
for epoch in range(500):
    # 1) Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x_data)

    # 2) Compute and print loss
    loss = criterion(y_pred, y_data)
    print(f'Epoch: {epoch} | Loss: {loss.item()} ')

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# %% colab={} colab_type="code" id="a_XD8DN2n9ob" outputId="f3e361f3-85cf-4093-bc66-c42bd4a8e8b9"
# After training
hour_var = tensor([[4.0]])
y_pred = model(hour_var)
print("Prediction (after training)", 4, model(hour_var).data[0][0].item())
