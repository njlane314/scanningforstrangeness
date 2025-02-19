import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import torch

class Visualiser:
    def __init__(self, dataset, planes=["U", "V", "W"], width=512, height=512):
        self.dataset = dataset
        self.planes = planes
        self.width, self.height = width, height

        try:
            self.run_arr = self.dataset.tree["run"].array(library="np")
            self.subrun_arr = self.dataset.tree["subrun"].array(library="np")
            self.event_arr = self.dataset.tree["event"].array(library="np")
        except Exception:
            self.run_arr = self.subrun_arr = self.event_arr = None

    def visualise_plane(self, idx, plane_idx, overlay=False, save=True, show=False):
        if idx >= len(self.dataset):
            print(f"Skipping event {idx}: Index out of range")
            return

        ev = self.dataset[idx]
        if isinstance(ev, tuple):
            img, truth = ev
            if img.ndim == 4 and img.shape[1] == 1:
                img = img.squeeze(1)
            if truth.ndim == 4 and truth.shape[1] == 1:
                truth = truth.squeeze(1)
            img, truth = img.numpy(), truth.numpy()
        else:
            if ev.ndim == 4 and ev.shape[1] == 1:
                ev = ev.squeeze(1)
            img = ev.numpy()
            truth = None

        if self.run_arr is not None:
            run = self.run_arr[idx]
            subrun = self.subrun_arr[idx]
            evt_num = self.event_arr[idx]
        else:
            run, subrun, evt_num = "Unknown", "Unknown", idx

        print(f"Rendering Event {idx} | Plane {self.planes[plane_idx]} | Run {run}, Subrun {subrun}, Event {evt_num}")

        fig, ax = plt.subplots(figsize=(12, 12), dpi=600)
        plane_img = img[plane_idx]
        norm = colors.PowerNorm(gamma=0.35, vmin=plane_img.min(), vmax=plane_img.max())
        ax.imshow(plane_img, origin="lower", cmap="jet", norm=norm)
        if overlay and truth is not None:
            ax.imshow(truth[plane_idx], origin="lower", cmap="cool", alpha=0.4)

        ax.set_xticks([0, self.width - 1])
        ax.set_yticks([0, self.height - 1])
        ax.tick_params(axis="both", direction="out", length=6, width=1.5, labelsize=18)
        ax.set_xlim(0, self.width - 1)
        ax.set_ylim(0, self.height - 1)
        ax.set_xlabel("Local Wire Coord", fontsize=20)
        ax.set_ylabel("Local Drift Time", fontsize=20)
        ax.set_title(f"Plane {self.planes[plane_idx]} (Run {run}, Subrun {subrun}, Event {evt_num})", fontsize=22)

        plt.tight_layout()
        if save:
            plt.savefig(f"event_{run}_{subrun}_{evt_num}_plane_{self.planes[plane_idx]}.png")
        if show:
            plt.show()
        plt.close(fig)

    def visualise_event_planes(self, idx, overlay=False, save=True, show=False):
        for i in range(len(self.planes)):
            self.visualise_plane(idx, plane_idx=i, overlay=overlay, save=save, show=show)