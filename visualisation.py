from dataclasses import dataclass
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patches as mpatches
import numpy as np
import torch

@dataclass
class VisualizationConfig:
    FIGURE_SIZE: Tuple[int, int] = (12, 12)
    DPI: int = 600
    GAMMA: float = 0.35
    PLANE_NAMES: List[str] = field(default_factory=lambda: ["U", "V", "W"])
    OVERLAY_COLORS: Dict[int, str] = field(default_factory=lambda: {
        1: "#00FFFF",  # Cyan
        2: "#FF00FF",  # Magenta
        3: "#FFFF00",  # Yellow
        4: "#00FF00",  # Lime Green
        5: "#FFA500",  # Orange
        6: "#FF0000"   # Red
    })
    LEGEND_LABELS: Dict[int, str] = field(default_factory=lambda: {
        1: "Noise",
        2: "Muon",
        3: "Charged Kaon",
        4: "Kaon Short",
        5: "Lambda",
        6: "Charged Sigma"
    })

class Visualiser:
    def __init__(self, config: Dict, vis_config: VisualizationConfig = VisualizationConfig()):
        self.seg_classes = config.get("train.segmentation_classes")
        self.width = config.get("dataset.width")
        self.height = config.get("dataset.height")
        self.vis_config = vis_config

    def _random_event_index(self, dataset) -> int:
        sig_indices = [i for i, t in enumerate(dataset.type_array) if t == 0]
        return np.random.choice(sig_indices if sig_indices else range(len(dataset)))

    def _setup_axes(self, ax: plt.Axes) -> None:
        ax.set_xticks([0, self.width - 1])
        ax.set_yticks([0, self.height - 1])
        ax.tick_params(axis="both", direction="out", length=6, width=1.5, labelsize=18)
        ax.set_xlim(0, self.width - 1)
        ax.set_ylim(0, self.height - 1)
        ax.set_xlabel("Local Drift Time", fontsize=20)
        ax.set_ylabel("Local Wire Coord", fontsize=20)

    def _get_seg_mask(self, truth_img: torch.Tensor, plane_idx: int) -> np.ndarray:
        start_idx = plane_idx * self.seg_classes
        end_idx = start_idx + self.seg_classes
        plane_truth = truth_img[start_idx:end_idx]
        
        if torch.is_tensor(plane_truth):
            return plane_truth.argmax(dim=0).cpu().numpy()
        return plane_truth.argmax(axis=0)

    def _create_figure(self, title: str) -> Tuple[plt.Figure, plt.Axes]:
        fig, ax = plt.subplots(figsize=self.vis_config.FIGURE_SIZE, 
                             dpi=self.vis_config.DPI)
        ax.set_title(title, fontsize=22)
        self._setup_axes(ax)
        return fig, ax

    def _save_and_close(self, fig: plt.Figure, filename: str) -> None:
        plt.tight_layout()
        plt.savefig(filename)
        plt.close(fig)

    def visualise_input_event(self, dataset) -> None:
        input_img, _, r, sr, evnum = dataset[self._random_event_index(dataset)]
        planes = self.vis_config.PLANE_NAMES[:input_img.shape[0]]

        for i, plane in enumerate(planes):
            fig, ax = self._create_figure(f"Plane {plane} (Run {r}, Subrun {sr}, Event {evnum})")
            ax.imshow(input_img[i], origin="lower", cmap="jet",
                     norm=colors.PowerNorm(gamma=self.vis_config.GAMMA,
                                         vmin=input_img.min(),
                                         vmax=input_img.max()))
            self._save_and_close(fig, f"event_{r}_{sr}_{evnum}_plane_{plane}.png")

    def visualise_truth_event(self, dataset) -> None:
        _, truth_img, r, sr, evnum = dataset[self._random_event_index(dataset)]
        num_planes = truth_img.shape[0] // self.seg_classes
        planes = self.vis_config.PLANE_NAMES[:num_planes]

        for i, plane in enumerate(planes):
            seg_mask = self._get_seg_mask(truth_img, i)
            fig, ax = self._create_figure(f"Plane {plane} (Run {r}, Subrun {sr}, Event {evnum})")
            im = ax.imshow(seg_mask, origin="lower", cmap="tab10", interpolation="nearest")
            plt.colorbar(im, ax=ax)
            self._save_and_close(fig, f"truth_event_{r}_{sr}_{evnum}_plane_{plane}.png")

    def visualise_overlay_event(self, dataset) -> None:
        input_img, truth_img, r, sr, evnum = dataset[self._random_event_index(dataset)]
        planes = self.vis_config.PLANE_NAMES[:input_img.shape[0]]

        class_colors = [(0, 0, 0, 0) if c == 0 else 
                       colors.to_rgba(self.vis_config.OVERLAY_COLORS.get(c, "#FFFFFF"), 1.0)
                       for c in range(self.seg_classes)]

        for i, plane in enumerate(planes):
            seg_mask = self._get_seg_mask(truth_img, i)
            fig, ax = self._create_figure(f"Plane {plane} (Run {r}, Subrun {sr}, Event {evnum})")
            
            ax.imshow(input_img[i], origin="lower", cmap="jet",
                     norm=colors.PowerNorm(gamma=self.vis_config.GAMMA,
                                         vmin=input_img.min(),
                                         vmax=input_img.max()))
            
            overlay = np.array(class_colors)[seg_mask]
            overlay[..., 3] = np.where(seg_mask != 0, 0.5, 0.0)
            ax.imshow(overlay, origin="lower", interpolation="nearest")

            legend_items = [(mpatches.Circle((0, 0), radius=5, facecolor=class_colors[c], linewidth=0),
                           self.vis_config.LEGEND_LABELS.get(c, str(c)))
                           for c in range(self.seg_classes) if np.any(seg_mask == c) and c != 0]
            
            if legend_items:
                handles, labels = zip(*legend_items)
                legend = ax.legend(handles, labels, loc='upper left', fontsize=18,
                                 frameon=False, handlelength=1.5, handletextpad=0.5,
                                 labelspacing=0.5)
                for text in legend.get_texts():
                    text.set_color("white")

            self._save_and_close(fig, f"overlay_event_{r}_{sr}_{evnum}_plane_{plane}.png")