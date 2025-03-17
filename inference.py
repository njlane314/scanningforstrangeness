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
    def __init__(self, args, file):
        self.args = args
        self.file_path = file
        self.tree_name = "imageanalyser/ImageTree"
        self.img_size = args.img_size
        self.num_classes = args.num_classes
        self.num_planes = args.num_planes
        self.enum_to_model = {val: idx for idx, val in enumerate(args.target_labels)}
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
        images = []
        for plane in range(self.num_planes):
            plane_iterable = data["input"][0][plane]
            plane_array = np.fromiter(plane_iterable, dtype=np.float32, count=self.img_size * self.img_size)
            images.append(plane_array.reshape(self.img_size, self.img_size))
        images_tensor = torch.tensor(np.stack(images), dtype=torch.float32)
        labels = []
        for plane_idx in range(self.num_planes):
            truth_iterable = data["truth"][0][plane_idx]
            truth_array = np.fromiter(truth_iterable, dtype=np.int64, count=self.img_size * self.img_size)
            labels.append(truth_array.reshape(self.img_size, self.img_size))
        label_tensor = torch.tensor(np.stack(labels), dtype=torch.long)
        remapped_labels = label_tensor.clone()
        for enum_val, model_val in self.enum_to_model.items():
            remapped_labels[label_tensor == enum_val] = model_val
        mask = ~torch.isin(label_tensor, torch.tensor(list(self.enum_to_model.keys()), dtype=torch.long))
        if mask.any():
            remapped_labels[mask] = 1  
        return images_tensor, remapped_labels, run, subrun, event

@dataclass
class VisualizationConfig:
    FIGURE_SIZE: Tuple[int, int] = (12, 12)
    DPI: int = 600
    GAMMA: float = 0.35
    PLANE_NAMES: List[str] = None  
    OVERLAY_COLORS: Dict[int, str] = None  
    LEGEND_LABELS: Dict[int, str] = None  

class Visualiser:
    def __init__(self, num_classes: int, width: int, height: int, target_labels: List[int], vis_config: VisualizationConfig):
        self.num_classes = num_classes
        self.width = width
        self.height = height
        self.target_labels = target_labels
        self.vis_config = vis_config
        self.model_to_enum = {idx: val for idx, val in enumerate(target_labels)}
        self.enum_to_model = {val: idx for idx, val in enumerate(target_labels)}
        self.vis_config.PLANE_NAMES = ["U", "V", "W"][:3]  
        self.vis_config.OVERLAY_COLORS = {
            1: "#00FFFF",  # Cyan
            2: "#FF00FF",  # Magenta
            3: "#FFFF00",  # Yellow
            4: "#00FF00",  # Lime Green
            5: "#FFA500",  # Orange
            6: "#FF0000"   # Red
        }
        self.vis_config.LEGEND_LABELS = {
            1: "Background",
            2: "Muon",
            3: "Charged Kaon",
            4: "Kaon Short",
            5: "Lambda",
            6: "Charged Sigma"
        }
        self.vis_config.OVERLAY_COLORS = {k: v for k, v in self.vis_config.OVERLAY_COLORS.items() if k in [self.enum_to_model.get(t, 0) for t in target_labels] or k == 0}
        self.vis_config.LEGEND_LABELS = {k: v for k, v in self.vis_config.LEGEND_LABELS.items() if k in [self.enum_to_model.get(t, 0) for t in target_labels] or k == 0}

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

    def visualise_prediction(self, dataset, model, device, output_dir: str):
        model.eval()
        os.makedirs(output_dir, exist_ok=True)
        images, truth, r, sr, evnum = dataset[self._random_event_index(dataset)]
        images = images.unsqueeze(0).to(device) 
        with torch.no_grad():
            pred = model(images)
            pred = pred.argmax(dim=2).squeeze(0).cpu().numpy()  # [num_planes, H, W]
        truth = truth.numpy()
        input_img = images.squeeze(0).cpu().numpy()

        planes = self.vis_config.PLANE_NAMES[:input_img.shape[0]]
        pred_enum = np.vectorize(self.model_to_enum.get)(pred)
        truth_enum = truth  

        for i, plane in enumerate(planes):
            fig, ax = self._create_figure(f"Plane {plane} Input (Run {r}, Subrun {sr}, Event {evnum})")
            ax.imshow(input_img[i], origin="lower", cmap="jet",
                      norm=colors.PowerNorm(gamma=self.vis_config.GAMMA,
                                           vmin=input_img.min(),
                                           vmax=input_img.max()))
            self._save_and_close(fig, os.path.join(output_dir, f"input_{r}_{sr}_{evnum}_plane_{plane}.png"))

            fig, ax = self._create_figure(f"Plane {plane} Prediction (Run {r}, Subrun {sr}, Event {evnum})")
            ax.imshow(input_img[i], origin="lower", cmap="jet",
                      norm=colors.PowerNorm(gamma=self.vis_config.GAMMA,
                                           vmin=input_img.min(),
                                           vmax=input_img.max()))
            class_colors = [(0, 0, 0, 0) if c not in self.vis_config.OVERLAY_COLORS else
                            colors.to_rgba(self.vis_config.OVERLAY_COLORS[c], 1.0)
                            for c in range(max(self.vis_config.OVERLAY_COLORS.keys()) + 1)]
            overlay = np.array(class_colors)[pred[i]]
            overlay[..., 3] = np.where(pred[i] != 0, 0.5, 0.0)
            ax.imshow(overlay, origin="lower", interpolation="nearest")
            legend_items = [(mpatches.Circle((0, 0), radius=5, facecolor=class_colors[c], linewidth=0),
                            self.vis_config.LEGEND_LABELS[c])
                            for c in np.unique(pred[i]) if c != 0 and c in self.vis_config.LEGEND_LABELS]
            if legend_items:
                handles, labels = zip(*legend_items)
                legend = ax.legend(handles, labels, loc='upper left', fontsize=18,
                                   frameon=False, handlelength=1.5, handletextpad=0.5,
                                   labelspacing=0.5)
                for text in legend.get_texts():
                    text.set_color("white")
            self._save_and_close(fig, os.path.join(output_dir, f"pred_{r}_{sr}_{evnum}_plane_{plane}.png"))

            fig, ax = self._create_figure(f"Plane {plane} Truth (Run {r}, Subrun {sr}, Event {evnum})")
            ax.imshow(input_img[i], origin="lower", cmap="jet",
                      norm=colors.PowerNorm(gamma=self.vis_config.GAMMA,
                                           vmin=input_img.min(),
                                           vmax=input_img.max()))
            overlay = np.array(class_colors)[truth[i]]
            overlay[..., 3] = np.where(truth[i] != 0, 0.5, 0.0)
            ax.imshow(overlay, origin="lower", interpolation="nearest")
            legend_items = [(mpatches.Circle((0, 0), radius=5, facecolor=class_colors[c], linewidth=0),
                            self.vis_config.LEGEND_LABELS[c])
                            for c in np.unique(truth[i]) if c != 0 and c in self.vis_config.LEGEND_LABELS]
            if legend_items:
                handles, labels = zip(*legend_items)
                legend = ax.legend(handles, labels, loc='upper left', fontsize=18,
                                   frameon=False, handlelength=1.5, handletextpad=0.5,
                                   labelspacing=0.5)
                for text in legend.get_texts():
                    text.set_color("white")
            self._save_and_close(fig, os.path.join(output_dir, f"truth_{r}_{sr}_{evnum}_plane_{plane}.png"))

def get_parser():
    parser = argparse.ArgumentParser(description="Inference and Visualisation Script")
    parser.add_argument("--model-path", type=str, default="/gluster/data/dune/niclane/checkpoints/segmentation/unet_epoch0_batch200_20250314_135526_torchscript.pt", 
                        help="Path to the saved TorchScript model")
    parser.add_argument("--labels-path", type=str, default="/gluster/data/dune/niclane/checkpoints/segmentation/target_labels_epoch0_batch200_20250314_135526.pth", 
                        help="Path to the saved target labels .pth file")
    parser.add_argument("--root-file", type=str, default="/gluster/data/dune/niclane/signal/nlane_prod_strange_resample_fhc_run2_fhc_reco2_reco2_trainingimage_signal_lambdamuon_1000_ana.root",
                        help="Path to the input ROOT file")
    parser.add_argument("--img-size", default=512, type=int, help="Square image dimension in pixels")
    parser.add_argument("--num-planes", default=3, type=int, help="Number of image planes")
    parser.add_argument("--output-dir", type=str, default="./event_displays", help="Directory to save event displays")
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()

    target_labels = torch.load(args.labels_path)
    args.num_classes = len(target_labels)
    args.target_labels = target_labels

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.jit.load(args.model_path, map_location=device)
    model.eval()

    dataset = ImageDataset(args, args.root_file)
    vis_config = VisualizationConfig()
    visualiser = Visualiser(
        num_classes=args.num_classes,
        width=args.img_size,
        height=args.img_size,
        target_labels=args.target_labels,
        vis_config=vis_config
    )

    visualiser.visualise_prediction(dataset, model, device, args.output_dir)
    print(f"Event displays saved to {args.output_dir}")

if __name__ == "__main__":
    main()