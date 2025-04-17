import argparse
import os
import uproot
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patches as mpatches
from dataclasses import dataclass
from typing import Dict, List, Tuple

class ImageDataset:
    def __init__(self, args, file, plane, foreground_labels):
        self.args = args
        self.file_path = file
        self.tree_name = "imageanalyser/ImageTree"
        self.img_size = args.img_size
        self.plane = plane
        self.foreground_labels = foreground_labels
        self.num_classes = len(foreground_labels)
        self.root_file = uproot.open(self.file_path, array_cache=None, num_workers=0)
        self.tree = self.root_file[self.tree_name]
        event_types = self.tree["type"].array(library="np")
        self.indices = np.where(event_types == 0)[0]
        self.num_events = len(self.indices)

    def __len__(self):
        return self.num_events

    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        data = self.tree.arrays(
            ["input", "truth", "run", "subrun", "event"],
            entry_start=actual_idx, entry_stop=actual_idx + 1,
            library="np"
        )
        run = data["run"][0]
        subrun = data["subrun"][0]
        event = data["event"][0]
        plane_iterable = data["input"][0][self.plane]
        plane_array = np.fromiter(plane_iterable, dtype=np.float32, count=self.img_size * self.img_size)
        images_tensor = torch.tensor(plane_array.reshape(1, self.img_size, self.img_size), dtype=torch.float32)
        truth_iterable = data["truth"][0][self.plane]
        truth_array = np.fromiter(truth_iterable, dtype=np.int64, count=self.img_size * self.img_size)
        truth_tensor = torch.tensor(truth_array.reshape(self.img_size, self.img_size), dtype=torch.long)
        masks = torch.zeros(self.num_classes, self.img_size, self.img_size, dtype=torch.float32)
        for i, label in enumerate(self.foreground_labels):
            masks[i] = (truth_tensor == label).float()
        return images_tensor, masks, run, subrun, event

@dataclass
class VisualizationConfig:
    FIGURE_SIZE: Tuple[int, int] = (12, 12)
    DPI: int = 600
    GAMMA: float = 0.35
    PLANE_NAMES: List[str] = None
    OVERLAY_COLORS: Dict[int, str] = None
    LEGEND_LABELS: Dict[int, str] = None

class Visualiser:
    def __init__(self, num_classes: int, width: int, height: int, foreground_labels: List[int], vis_config: VisualizationConfig):
        self.num_classes = num_classes
        self.width = width
        self.height = height
        self.foreground_labels = foreground_labels
        self.vis_config = vis_config
        self.vis_config.PLANE_NAMES = ["U", "V", "W"]
        color_list = ["#00FFFF", "#FF00FF", "#FFFF00", "#00FF00", "#FFA500", "#FF0000"]
        self.vis_config.OVERLAY_COLORS = {label: color_list[i % len(color_list)] for i, label in enumerate(foreground_labels)}
        self.vis_config.LEGEND_LABELS = {label: f"Label {label}" for label in foreground_labels}

    def _random_event_index(self, dataset) -> int:
        return np.random.randint(0, len(dataset))

    def _setup_axes(self, ax: plt.Axes) -> None:
        ax.set_xticks([0, self.width - 1])
        ax.set_yticks([0, self.height - 1])
        ax.tick_params(axis="both", direction="out", length=6, width=1.5, labelsize=18)
        ax.set_xlim(0, self.width - 1)
        ax.set_ylim(0, self.height - 1)
        ax.set_xlabel("Local Drift Time", fontsize=20)
        ax.set_ylabel("Local Wire Coord", fontsize=20)

    def _create_figure(self, title: str) -> Tuple[plt.Figure, plt.Axes]:
        fig, ax = plt.subplots(figsize=self.vis_config.FIGURE_SIZE, dpi=self.vis_config.DPI)
        ax.set_title(title, fontsize=22)
        self._setup_axes(ax)
        return fig, ax

    def _save_and_close(self, fig: plt.Figure, filename: str) -> None:
        plt.tight_layout()
        plt.savefig(filename)
        plt.close(fig)

    def visualise_prediction(self, dataset, model, device, output_dir: str, plane: int):
        model.eval()
        os.makedirs(output_dir, exist_ok=True)
        images, truth, r, sr, evnum = dataset[self._random_event_index(dataset)]
        images = images.unsqueeze(0).to(device)  
        with torch.no_grad():
            pred = model(images)
            pred_sigmoid = torch.sigmoid(pred)
            pred_binary = (pred_sigmoid > 0.95).cpu().numpy()  
        truth = truth.numpy()  
        input_img = images.squeeze(0).cpu().numpy()  

        plane_name = self.vis_config.PLANE_NAMES[plane]

        fig, ax = self._create_figure(f"Plane {plane_name} Input (Run {r}, Subrun {sr}, Event {evnum})")
        ax.imshow(input_img[0], origin="lower", cmap="jet",
                  norm=colors.PowerNorm(gamma=self.vis_config.GAMMA,
                                        vmin=input_img.min(),
                                        vmax=input_img.max()))
        self._save_and_close(fig, os.path.join(output_dir, f"input_{r}_{sr}_{evnum}_plane_{plane_name}.png"))

        fig, ax = self._create_figure(f"Plane {plane_name} Prediction (Run {r}, Subrun {sr}, Event {evnum})")
        ax.imshow(input_img[0], origin="lower", cmap="jet",
                  norm=colors.PowerNorm(gamma=self.vis_config.GAMMA,
                                        vmin=input_img.min(),
                                        vmax=input_img.max()))
        for i, label in enumerate(self.foreground_labels):
            mask = pred_binary[0, i, :, :]
            mask = mask.astype(bool)
            if np.any(mask):
                overlay = np.zeros((self.height, self.width, 4))
                color = colors.to_rgba(self.vis_config.OVERLAY_COLORS[label], 0.5)
                overlay[mask, :] = color
                ax.imshow(overlay, origin="lower", interpolation="nearest")
        legend_items = [(mpatches.Circle((0, 0), radius=5, facecolor=self.vis_config.OVERLAY_COLORS[label], linewidth=0),
                         self.vis_config.LEGEND_LABELS[label])
                        for i, label in enumerate(self.foreground_labels) if np.any(pred_binary[0, i, :, :])]
        if legend_items:
            handles, labels = zip(*legend_items)
            legend = ax.legend(handles, labels, loc='upper left', fontsize=18,
                               frameon=False, handlelength=1.5, handletextpad=0.5,
                               labelspacing=0.5)
            for text in legend.get_texts():
                text.set_color("white")
        self._save_and_close(fig, os.path.join(output_dir, f"pred_{r}_{sr}_{evnum}_plane_{plane_name}.png"))

        fig, ax = self._create_figure(f"Plane {plane_name} Truth (Run {r}, Subrun {sr}, Event {evnum})")
        ax.imshow(input_img[0], origin="lower", cmap="jet",
                  norm=colors.PowerNorm(gamma=self.vis_config.GAMMA,
                                        vmin=input_img.min(),
                                        vmax=input_img.max()))
        for i, label in enumerate(self.foreground_labels):
            mask = truth[i, :, :]
            mask = mask.astype(bool)
            if np.any(mask):
                overlay = np.zeros((self.height, self.width, 4))
                color = colors.to_rgba(self.vis_config.OVERLAY_COLORS[label], 0.5)
                overlay[mask, :] = color
                ax.imshow(overlay, origin="lower", interpolation="nearest")
        legend_items = [(mpatches.Circle((0, 0), radius=5, facecolor=self.vis_config.OVERLAY_COLORS[label], linewidth=0),
                         self.vis_config.LEGEND_LABELS[label])
                        for i, label in enumerate(self.foreground_labels) if np.any(truth[i, :, :])]
        if legend_items:
            handles, labels = zip(*legend_items)
            legend = ax.legend(handles, labels, loc='upper left', fontsize=18,
                               frameon=False, handlelength=1.5, handletextpad=0.5,
                               labelspacing=0.5)
            for text in legend.get_texts():
                text.set_color("white")
        self._save_and_close(fig, os.path.join(output_dir, f"truth_{r}_{sr}_{evnum}_plane_{plane_name}.png"))

def get_parser():
    parser = argparse.ArgumentParser(description="Inference and Visualisation Script")
    parser.add_argument("--model-path", type=str, default="/gluster/data/dune/niclane/checkpoints/segmentation/uresnet_plane0_9_20250321_135456_ts.pt",
                        help="Path to the saved TorchScript model (e.g., uresnet_plane0_..._ts.pt)")
    parser.add_argument("--labels-path", type=str, default="/gluster/data/dune/niclane/checkpoints/segmentation/label_mapping_plane0.npz",
                        help="Path to the saved label mapping .npz file (e.g., label_mapping_plane0.npz)")
    parser.add_argument("--root-file", type=str, default="/gluster/data/dune/niclane/signal/nlane_prod_strange_resample_fhc_run2_fhc_reco2_reco2_trainingimage_signal_lambdamuon_1000_ana.root",
                        help="Path to the input ROOT file")
    parser.add_argument("--img-size", default=512, type=int,
                        help="Square image dimension in pixels")
    parser.add_argument("--plane", type=int, choices=[0, 1, 2], required=True,
                        help="Plane number (0, 1, 2)")
    parser.add_argument("--output-dir", type=str, default="./event_displays",
                        help="Directory to save event displays")
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()

    label_mapping = np.load(args.labels_path)
    foreground_labels = [int(k) for k in label_mapping.keys()]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.jit.load(args.model_path, map_location=device)
    model.eval()

    dataset = ImageDataset(args, args.root_file, args.plane, foreground_labels)
    vis_config = VisualizationConfig()
    visualiser = Visualiser(
        num_classes=len(foreground_labels),
        width=args.img_size,
        height=args.img_size,
        foreground_labels=foreground_labels,
        vis_config=vis_config
    )

    visualiser.visualise_prediction(dataset, model, device, args.output_dir, args.plane)
    print(f"Event displays saved to {args.output_dir}")

if __name__ == "__main__":
    main()