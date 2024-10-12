import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from boring_utils.utils import cprint


# From t3-1
def visualize_gradients_weight(
        net, 
        train_set,
        device,
        color="C0", 
        norm_x_axis=True,
        norm_y_axis=False,
        print_variance=False):
    """
    Use cross entropy loss for the first batch of the training set to calculate the gradients of the weights.
    """
    
    net.eval()
    small_loader = DataLoader(train_set, batch_size=256, shuffle=False)
    imgs, labels = next(iter(small_loader))
    imgs, labels = imgs.to(device), labels.to(device)
    
    # Pass one batch through the network, and calculate the gradients for the weights
    net.zero_grad()
    preds = net(imgs)
    loss = F.cross_entropy(preds, labels)
    loss.backward()
    # We limit our visualization to the weight parameters and exclude the bias to reduce the number of plots
    grads = {name: params.grad.data.view(-1).cpu().clone().numpy() for name, params in net.named_parameters() if "weight" in name}
    net.zero_grad()

    # Calculate the min and max values for the x and y axis
    all_grads = np.concatenate(list(grads.values()))
    if norm_x_axis:
        x_min, x_max = np.min(all_grads), np.max(all_grads)
    if norm_y_axis:
        y_max = 0 
        for key in grads:
            counts, _ = np.histogram(grads[key], bins=30)
            y_max = max(y_max, np.max(counts))

    # Plotting
    columns = len(grads)
    fig, ax = plt.subplots(1, columns, figsize=(columns*3.5, 2.5))
    fig_index = 0
    for key in grads:
        key_ax = ax[fig_index % columns]
        sns.histplot(
            data=grads[key], bins=30, ax=key_ax, color=color, kde=True)
        key_ax.set_title(str(key), fontsize=16)

        if norm_x_axis: key_ax.set_xlim(x_min, x_max)
        key_ax.set_xlabel("Grad magnitude")
        key_ax.set_xticklabels(key_ax.get_xticklabels(), rotation=30)
        if norm_y_axis: key_ax.set_ylim(0, y_max)
        fig_index += 1

    fig.suptitle(f"Grad magnitude dist for act func {net.act_fn}", fontsize=24, y=1.2)
    fig.subplots_adjust(wspace=0.6)
    plt.show()
    plt.close() 

    if norm_x_axis: cprint(x_min, x_max)
    if norm_y_axis: cprint(y_max)
    if print_variance:
        for key in sorted(grads.keys()):
            print(f"{key} - Variance: {np.var(grads[key])}")


# From t3-2
def visualize_activations(
        net, 
        train_set,
        device,
        color="C0",
        norm_x_axis=True,
        norm_y_axis=False
    ):
    activations = {}
    
    net.eval()
    small_loader = DataLoader(train_set, batch_size=1024)
    imgs, labels = next(iter(small_loader))
    with torch.no_grad():
        layer_index = 0
        imgs = imgs.to(device)
        imgs = imgs.view(imgs.size(0), -1)
        # We need to manually loop through the layers to save all activations
        for layer_index, layer in enumerate(net.layers[:-1]):
            imgs = layer(imgs)
            activations[layer_index] = imgs.view(-1).cpu().numpy()
    
    # Calculate the min and max values for the x and y axis
    all_activations = np.concatenate(list(activations.values()))
    if norm_x_axis:
        x_min, x_max = np.min(all_activations), np.max(all_activations)
    if norm_y_axis:
        y_max = 0 
        for key in activations:
            counts, _ = np.histogram(activations[key], bins=50, density=True)
            y_max = max(y_max, np.max(counts))
    
    # Plotting
    columns = 4
    rows = int(np.ceil(len(activations)/columns))
    fig, ax = plt.subplots(rows, columns, figsize=(columns*2.7, rows*2.5))
    fig_index = 0
    for key in activations:
        key_ax = ax[fig_index//columns][fig_index%columns]
        sns.histplot(
            data=activations[key], bins=50, ax=key_ax, 
            color=color, kde=True, stat="density")
        key_ax.set_title(
            f"Layer {key} - {net.layers[key].__class__.__name__}", 
            fontsize=16)
        
        if norm_x_axis:
            key_ax.set_xlim(x_min, x_max)
        key_ax.set_xlabel("Activation value")
        if norm_y_axis:
            key_ax.set_ylim(0, y_max)
        
        fig_index += 1

    # Hide empty subplots
    for i in range(fig_index, rows * columns):
        ax[i//columns][i%columns].axis('off')

    fig.suptitle(f"Activation dist for act func {net.act_fn}", fontsize=24, y=1.05)
    fig.subplots_adjust(hspace=0.6, wspace=0.6)
    plt.show()
    plt.close() 

    if norm_x_axis: cprint(x_min, x_max)
    if norm_y_axis: cprint(y_max)