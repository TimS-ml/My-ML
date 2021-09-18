# linear model: y = x * w
# the actual w is 2
# [1] start with a random w
# [2] update w based on gradient (not random guess)
# - w requires_grad; loss.backward() -> update grad of w
# - update w

import torch

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]
w = torch.tensor([1.0], requires_grad=True)


# our model forward pass
def forward(x):
    return x * w


# Loss function
def loss(y_pred, y_val):
    return (y_pred - y_val)**2


# Before training
print("Prediction (before training)", 4, forward(4).item())

# Training loop
for epoch in range(10):
    for x_val, y_val in zip(x_data, y_data):
        y_pred = forward(x_val)
        l = loss(y_pred, y_val)
        l.backward()  # Back propagation to update weights
        print("\tgrad: ", x_val, y_val, w.grad.item())
        # update w based on grad
        # this will be replaced with 'optimizer.step()'
        w.data = w.data - 0.01 * w.grad.item()

        # Manually zero the gradients after updating weights
        w.grad.data.zero_()

    print(f"Epoch: {epoch} | Loss: {l.item()}")

# After training
print("Prediction (after training)", 4, forward(4).item())
