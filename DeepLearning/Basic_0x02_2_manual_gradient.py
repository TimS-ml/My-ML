# linear model: y = x * w
# the actual w is 2
# [1] start with a random w
# [2] update w based on gradient (not random guess)

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = 1.0  # a random guess: random value


# our model forward pass
def forward(x):
    return x * w


# Loss function
def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2


# compute gradient = d_loss/d_w
# loss = (x * w - y) ** 2
# https://en.wikipedia.org/wiki/Chain_rule
def gradient(x, y):
    return 2 * x * (x * w - y)


# The actual output should be 8
# Before training
print("Prediction (before training), x=4", forward(4))

# Training loop
for epoch in range(10):
    for x_val, y_val in zip(x_data, y_data):
        # Compute derivative w.r.t to the learned weights
        # Update the weights
        # Compute the loss and print progress
        grad = gradient(x_val, y_val)
        w = w - 0.01 * grad
        print("\tgrad: ", x_val, y_val, round(grad, 2))
        l = loss(x_val, y_val)
    print("progress:", epoch, "w=", round(w, 2), "loss=", round(l, 2))

# After training
print("Predicted score (after training), x=4", forward(4))
