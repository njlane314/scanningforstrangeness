import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import torch
import time
from matplotlib.colors import ListedColormap, BoundaryNorm

from lib.dataset import ImageDataLoader
from lib.config import ConfigLoader
from lib.model import UNet

def visualise_input(input_histogram, height, width):
    fig, ax = plt.subplots(figsize=(12, 12), dpi=300)
    img = ax.imshow(input_histogram, origin="lower", cmap='jet', norm=colors.PowerNorm(gamma=0.35, vmin=1e3, vmax=input_histogram.max()), aspect='equal', interpolation='none')
    ax.set_xticks([0, width - 1])
    ax.set_yticks([0, height - 1])
    ax.tick_params(axis="both", direction="out", length=6, width=1.5, labelsize=18)
    ax.set_xlim(0, width - 1)
    ax.set_ylim(0, height - 1)
    ax.set_xlabel('Drift Time', fontsize=20)
    ax.set_ylabel('Wire Coord', fontsize=20)
    plt.tight_layout()
    return fig

def visualise_truth(target_histogram, height, width):
    cmap = ListedColormap(['#ffffff', '#0000ff', '#ff0000'])
    bounds = [-0.5, 0.5, 1.5, 2.5]
    norm = BoundaryNorm(bounds, cmap.N)
    
    fig, ax = plt.subplots(figsize=(12, 12), dpi=300)
    masked_target_histogram = np.ma.masked_invalid(target_histogram)
    img = ax.imshow(masked_target_histogram, cmap=cmap, norm=norm, aspect='equal', interpolation='none')
    ax.set_xticks([0, width - 1])
    ax.set_yticks([0, height - 1])
    ax.tick_params(axis="both", direction="out", length=6, width=1.5, labelsize=18)
    ax.set_xlim(0, width - 1)
    ax.set_ylim(0, height - 1)
    ax.set_xlabel('Drift Time', fontsize=20)
    ax.set_ylabel('Wire Coord', fontsize=20)
    plt.tight_layout()
    return fig

def visualise_prediction(prediction_histogram, height, width):
    prediction_class = np.argmax(prediction_histogram, axis=0)
    
    cmap = ListedColormap(['#ffffff', '#ff7f00', '#00ff00'])
    bounds = [-0.5, 0.5, 1.5, 2.5]
    norm = BoundaryNorm(bounds, cmap.N)
    
    fig, ax = plt.subplots(figsize=(12, 12), dpi=300)
    masked_prediction_histogram = np.ma.masked_invalid(prediction_class)
    img = ax.imshow(masked_prediction_histogram, cmap=cmap, norm=norm, aspect='equal', interpolation='none')
    ax.set_xticks([0, width - 1])
    ax.set_yticks([0, height - 1])
    ax.tick_params(axis="both", direction="out", length=6, width=1.5, labelsize=18)
    ax.set_xlim(0, width - 1)
    ax.set_ylim(0, height - 1)
    ax.set_xlabel('Drift Time', fontsize=20)
    ax.set_ylabel('Wire Coord', fontsize=20)
    plt.tight_layout()
    return fig

def load_model(model_path, device):
    model = UNet(1, n_classes=3, depth=4, n_filters=16).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()  
    return model

def visualise(config_file, model_path, n_events=1, sig_filter=False):
    config = ConfigLoader(config_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    data_loader = ImageDataLoader(
        input_dir=config.input_dir,
        target_dir=config.target_dir,
        batch_size=1, 
        train_pct=config.train_pct,
        valid_pct=config.valid_pct,
        device=device
    )

    model = load_model(model_path, device)
    
    height, width = config.height, config.width
    plot_dir = os.path.join(os.getcwd(), "infer")
    os.makedirs(plot_dir, exist_ok=True)

    event_count = 0
    for i, (input_img, target_img) in enumerate(data_loader.train_dl):
        if event_count >= n_events:
            break

        input_img = input_img.squeeze().cpu().numpy()
        target_img = target_img.squeeze().cpu().numpy()

        unique_values, counts = np.unique(target_img, return_counts=True)
        value_counts = dict(zip(unique_values, counts))
        
        if sig_filter and value_counts.get(2, 0) == 0:
            print(f"Skipping event {i + 1} - no unique values of '2' > 0")
            continue
        
        print(f"Unique Values and Counts in Truth Histogram for Event {i + 1}: {value_counts}")

        input_img_tensor = torch.tensor(input_img).unsqueeze(0).unsqueeze(0).to(device) 
        with torch.no_grad():
            prediction = model(input_img_tensor).squeeze().cpu().numpy()  
        
        timestamp = int(time.time())
        
        input_fig = visualise_input(input_img, height, width)
        input_path = os.path.join(plot_dir, f"input_event_{timestamp}_{i + 1}.png")
        input_fig.savefig(input_path, dpi=300)
        plt.close(input_fig)

        target_fig = visualise_truth(target_img, height, width)
        target_path = os.path.join(plot_dir, f"target_event_{timestamp}_{i + 1}.png")
        target_fig.savefig(target_path, dpi=300)
        plt.close(target_fig)

        prediction_fig = visualise_prediction(prediction, height, width)
        prediction_path = os.path.join(plot_dir, f"prediction_event_{timestamp}_{i + 1}.png")
        prediction_fig.savefig(prediction_path, dpi=300)
        plt.close(prediction_fig)

        print(f"Saved input histogram to {input_path}")
        print(f"Saved target histogram to {target_path}")
        print(f"Saved prediction histogram to {prediction_path}")
        
        event_count += 1

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True)
    parser.add_argument('-m', '--model', type=str, required=True)
    parser.add_argument('-n', '--n_events', type=int, default=1)
    parser.add_argument('-f', '--filter', type=bool, default=False)
    args = parser.parse_args()
    
    visualise(args.config, args.model, args.n_events, args.filter)