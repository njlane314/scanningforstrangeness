import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patches as mpatches
import numpy as np
import random
import torch
from scipy.ndimage import gaussian_filter
mpl.rcParams['text.usetex'] = True

class Visualiser:
    def __init__(self, config):
        self.seg_classes = config.get("model.seg_classes")
        self.width = config.get("dataset.width")
        self.height = config.get("dataset.height")
    def _get_random(self, dataset):
        sig = [i for i, t in enumerate(dataset.type) if t == 0]
        if not sig:
            return random.choice(dataset)
        return random.choice(sig)
    def visualise_input_event(self, dataset):
        idx = self._get_random(dataset)
        event_data = dataset[idx]
        input_img, _, r, sr, evnum = event_data
        planes = ["U", "V", "W"][:input_img.shape[0]]
        for i in range(input_img.shape[0]):
            fig, ax = plt.subplots(figsize=(12, 12), dpi=600)
            ax.imshow(input_img[i],
                      origin="lower",
                      cmap="jet",
                      norm=colors.PowerNorm(gamma=0.35, vmin=input_img.min(), vmax=input_img.max()))
            ax.set_xticks([0, self.width - 1])
            ax.set_yticks([0, self.height - 1])
            ax.tick_params(axis="both", direction="out", length=6, width=1.5, labelsize=18)
            ax.set_xlim(0, self.width - 1)
            ax.set_ylim(0, self.height - 1)
            ax.set_xlabel("Local Drift Time", fontsize=20)
            ax.set_ylabel("Local Wire Coord", fontsize=20)
            ax.set_title(f"Plane {planes[i]} (Run {r}, Subrun {sr}, Event {evnum})", fontsize=22)
            plt.tight_layout()
            plt.savefig(f"event_{r}_{sr}_{evnum}_plane_{planes[i]}.png")
            plt.close(fig)
    def visualise_truth_event(self, dataset):
        idx = self._get_random(dataset)
        event_data = dataset[idx]
        _, truth_img, r, sr, evnum = event_data
        num_planes = truth_img.shape[0] // self.seg_classes
        planes = ["U", "V", "W"][:num_planes]
        for i in range(num_planes):
            if torch.is_tensor(truth_img):
                plane_truth = truth_img[i * self.seg_classes:(i + 1) * self.seg_classes]
                seg_mask = plane_truth.argmax(dim=0).cpu().numpy()
            else:
                plane_truth = truth_img[i * self.seg_classes:(i + 1) * self.seg_classes]
                seg_mask = plane_truth.argmax(axis=0)
            fig, ax = plt.subplots(figsize=(12, 12), dpi=600)
            im = ax.imshow(seg_mask,
                           origin="lower",
                           cmap="tab10",
                           interpolation="nearest")
            ax.set_xticks([0, self.width - 1])
            ax.set_yticks([0, self.height - 1])
            ax.tick_params(axis="both", direction="out", length=6, width=1.5, labelsize=18)
            ax.set_xlim(0, self.width - 1)
            ax.set_ylim(0, self.height - 1)
            ax.set_xlabel("Local Drift Time", fontsize=20)
            ax.set_ylabel("Local Wire Coord", fontsize=20)
            ax.set_title(f"Plane {planes[i]} (Run {r}, Subrun {sr}, Event {evnum})", fontsize=22)
            plt.colorbar(im, ax=ax)
            plt.tight_layout()
            plt.savefig(f"truth_event_{r}_{sr}_{evnum}_plane_{planes[i]}.png")
            plt.close(fig)
    def visualise_overlay_event(self, dataset):
        idx = self._get_random(dataset)
        event_data = dataset[idx]
        input_img, truth_img, r, sr, evnum = event_data
        planes = ["U", "V", "W"][:input_img.shape[0]]
        custom_overlay_colors = {
            1: "#00FFFF", # Cyan
            2: "#FF00FF",  # Magenta
            3: "#FFFF00",  # Yellow
            4: "#00FF00",  # Lime Green
            5: "#FFA500",  # Orange
            6: "#FF0000"   # Red
        }
        class_colors = []
        for c in range(self.seg_classes):
            if c == 0:
                class_colors.append((0, 0, 0, 0))
            else:
                hex_color = custom_overlay_colors.get(c, "#FFFFFF")
                class_colors.append(colors.to_rgba(hex_color, alpha=1.0))
        legend_labels = {
            1: r"$\text{Noise}$",
            2: r"$\mu$",                   # Muon
            3: r"$K^{\pm}$",               # Charged Kaon
            4: r"$K^{0}_{S}$",             # Kaon Short
            5: r"$\Lambda$",               # Lambda
            6: r"$\Sigma^{\pm}$"           # Charged Sigma
        }
        for i in range(input_img.shape[0]):
            if torch.is_tensor(truth_img):
                plane_truth = truth_img[i * self.seg_classes:(i + 1) * self.seg_classes]
                seg_mask = plane_truth.argmax(dim=0).cpu().numpy()
            else:
                plane_truth = truth_img[i * self.seg_classes:(i + 1) * self.seg_classes]
                seg_mask = plane_truth.argmax(axis=0)
            fig, ax = plt.subplots(figsize=(12, 12), dpi=600)
            ax.imshow(input_img[i],
                      origin="lower",
                      cmap="jet",
                      norm=colors.PowerNorm(gamma=0.35,
                                            vmin=input_img.min(),
                                            vmax=input_img.max()))
            class_colors_arr = np.array(class_colors)  
            overlay = class_colors_arr[seg_mask].copy()  
            overlay[..., 3] = np.where(seg_mask != 0, 0.5, 0.0)
            ax.imshow(overlay, origin="lower", interpolation="nearest")
            legend_items = []
            for c in range(self.seg_classes):
                if np.any(seg_mask == c) and c != 0:
                    marker = mpatches.Circle((0, 0), radius=5,
                                             facecolor=class_colors[c],
                                             linewidth=0)
                    legend_items.append((marker, legend_labels.get(c, str(c))))
            if legend_items:
                handles, labels = zip(*legend_items)
                legend = ax.legend(handles, labels, loc='upper left', fontsize=18, frameon=False,
                                   handlelength=1.5, handletextpad=0.5, labelspacing=0.5)
                for text in legend.get_texts():
                    text.set_color("white")
            ax.set_xticks([0, self.width - 1])
            ax.set_yticks([0, self.height - 1])
            ax.tick_params(axis="both", direction="out", length=6, width=1.5, labelsize=18)
            ax.set_xlim(0, self.width - 1)
            ax.set_ylim(0, self.height - 1)
            ax.set_xlabel("Local Drift Time", fontsize=20)
            ax.set_ylabel("Local Wire Coord", fontsize=20)
            ax.set_title(f"Plane {planes[i]} (Run {r}, Subrun {sr}, Event {evnum})", fontsize=22)
            plt.tight_layout()
            plt.savefig(f"overlay_event_{r}_{sr}_{evnum}_plane_{planes[i]}.png")
            plt.close(fig)