import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

def visualise_truth(target_histogram, title=""):
    cmap = ListedColormap(['grey', 'red', 'blue', 'cyan'])
    bounds = [0, 1, 2, 3, 4]
    norm = BoundaryNorm(bounds, cmap.N)

    plt.figure(figsize=(10, 6))
    masked_target_histogram = np.ma.masked_invalid(target_histogram)
    plt.imshow(masked_target_histogram, cmap=cmap, norm=norm, aspect='equal', interpolation='none')

    cbar = plt.colorbar(ticks=[1.5, 2.5, 3.5])
    cbar.ax.set_yticklabels([r'$\mu$', r'$\pi^+$', r'$\pi^-$'])

    plt.xlabel("Drift Coordinate")
    plt.ylabel("Wire Coordinate")

    if title:
        plt.title(title)

    plt.tight_layout()
    plt.show()

def visualise_truth(input_histogram, threshold=1e3, title=""):
    input_histogram[input_histogram < threshold] = threshold

    plt.figure(figsize=(8, 6))
    plt.imshow(input_histogram, cmap='jet', norm=LogNorm(vmin=threshold), aspect='equal', interpolation='none')

    cbar = plt.colorbar()
    cbar.set_label("Charge")

    plt.xlabel("Drift Coordinate")
