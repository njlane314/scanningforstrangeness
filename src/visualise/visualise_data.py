import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm, LogNorm

def visualise_input(input_histogram, event_number):
    input_histogram[input_histogram < 1e3] = 1e3
    plt.figure(figsize=(8, 6))
    plt.imshow(input_histogram, cmap='jet', norm=LogNorm(vmin=1e3), aspect='equal', interpolation='none')
    cbar = plt.colorbar()
    cbar.set_label("Charge")
    plt.xlabel("Drift")
    plt.ylabel("Wire")
    plt.tight_layout()
    output_file = os.path.join("results", "plots", "input", f"input_event_{event_number}.png")
    plt.savefig(output_file, dpi=300)
    plt.close()

def visualise_truth(target_histogram, event_number):
    cmap = ListedColormap(['white', 'red', 'blue', 'cyan', 'green', 'yellow', 'purple'])
    bounds = np.arange(0, target_histogram.max() + 2)
    norm = BoundaryNorm(bounds, cmap.N)
    plt.figure(figsize=(8, 6))
    masked_target_histogram = np.ma.masked_invalid(target_histogram)
    plt.imshow(masked_target_histogram, cmap=cmap, norm=norm, aspect='equal', interpolation='none')
    plt.xlabel("Drift")
    plt.ylabel("Wire")
    plt.tight_layout()
    output_file = os.path.join("results", "plots", "target", f"target_event_{event_number}.png")
    plt.savefig(output_file, dpi=300)
    plt.close()

def load_histograms(input_folder, event_number):
    input_file = os.path.join(input_folder, "input", f"image_{event_number}.npz")
    target_file = os.path.join(input_folder, "target", f"image_{event_number}.npz")
    if not os.path.exists(input_file) or not os.path.exists(target_file):
        print(f"Histograms for event {event_number} not found.")
        return None, None
    input_histogram = np.load(input_file)['arr_0']
    target_histogram = np.load(target_file)['arr_0']
    return input_histogram, target_histogram

def visualise_event(input_folder, event_number):
    input_histogram, target_histogram = load_histograms(input_folder, event_number)
    if input_histogram is not None and target_histogram is not None:
        visualise_input(input_histogram, event_number)
        visualise_truth(target_histogram, event_number)
    else:
        print(f"Skipping visualisation for event {event_number}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--proc', type=str, required=True)
    parser.add_argument('-v', '--view', type=str, default="u", choices=["u", "v", "w"])
    parser.add_argument('-e', '--event', type=int, required=True)
    args = parser.parse_args()

    os.makedirs("results/plots/input", exist_ok=True)
    os.makedirs("results/plots/target", exist_ok=True)

    input_folder = os.path.join(args.proc, f"images_{args.view}")
    visualise_event(input_folder, args.event)
