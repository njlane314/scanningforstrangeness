import argparse
import os
import uproot
import numpy as np
import torch
import ROOT
from dataclasses import dataclass
from typing import Dict, List

ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetPalette(57)

class ImageDataset:
    def __init__(self, args, file, foreground_labels):
        self.args = args
        self.file_path = file
        self.tree_name = "imageanalyser/ImageTree"  
        self.plane = args.plane
        self.img_size = args.img_size
        self.foreground_labels = foreground_labels
        self.root_file = uproot.open(self.file_path, array_cache=None, num_workers=0)
        self.tree = self.root_file[self.tree_name]
        event_types = self.tree["type"].array(library="np")
        self.indices = np.where(event_types == 0)[0]  
        self.num_events = len(self.indices)
    def __len__(self):
        return self.num_events
    def get_full_event(self, idx):
        actual_idx = self.indices[idx]
        data = self.tree.arrays(
            ["input", "truth", "run", "subrun", "event"],
            entry_start=actual_idx, 
            entry_stop=actual_idx + 1,
            library="np"
        )
        img = np.fromiter(data["input"][0][self.plane], dtype=np.float32).reshape(1, self.img_size, self.img_size)
        truth = np.fromiter(data["truth"][0][self.plane], dtype=np.int64).reshape(self.img_size, self.img_size)
        run = data["run"][0]
        subrun = data["subrun"][0]
        event = data["event"][0]
        return img, truth, run, subrun, event

@dataclass
class VisualisationConfig:
    PLANE_NAMES: List[str] = None
    OVERLAY_COLORS: Dict[int, str] = None
    LEGEND_LABELS: Dict[int, str] = None

class Visualiser:
    def __init__(self, num_classes: int, width: int, height: int, foreground_labels: List[int], vis_config: VisualisationConfig):
        self.num_classes = num_classes
        self.width = width
        self.height = height
        self.foreground_labels = foreground_labels
        self.vis_config = vis_config
        self.vis_config.PLANE_NAMES = ["U", "V", "W"]
        color_list = ["#00FFFF", "#FF00FF", "#FFFF00", "#00FF00", "#FFA500", "#FF0000"]
        self.vis_config.OVERLAY_COLORS = {label: color_list[i % len(color_list)] for i, label in enumerate(foreground_labels)}
        self.vis_config.LEGEND_LABELS = {
            1: "Noise", 2: "Muon", 3: "Charged Kaon", 4: "Kaon Short", 5: "Lambda", 6: "Charged Sigma"
        }

    def _random_event_index(self, dataset) -> int:
        return np.random.randint(0, len(dataset))
    
    def visualise_truth(self, truth_data, plane, r, sr, evnum, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        seg_mask = truth_data.astype(np.int64)  # Shape: (img_size, img_size)
        plane_name = self.vis_config.PLANE_NAMES[plane]  # "U", "V", or "W"
        title = f"Plane {plane_name} Truth (Run {r}, Subrun {sr}, Event {evnum})"

        c_truth = ROOT.TCanvas(f"c_truth_{plane}", title, 1200, 1200)
        c_truth.SetFillColor(ROOT.kWhite)
        c_truth.SetMargin(0.08, 0.08, 0.08, 0.08)

        h_truth = ROOT.TH2F(f"h_truth_{plane}", title, self.width, 0, self.width, self.height, 0, self.height)
        for i in range(self.width):
            for j in range(self.height):
                h_truth.SetBinContent(i + 1, j + 1, seg_mask[j, i])

        h_truth.GetXaxis().SetTitle("Local Drift Time")
        h_truth.GetYaxis().SetTitle("Local Wire Coord")
        h_truth.GetXaxis().SetTitleOffset(1.0)
        h_truth.GetYaxis().SetTitleOffset(1.0)
        h_truth.GetXaxis().SetLabelSize(0.03)
        h_truth.GetYaxis().SetLabelSize(0.03)
        h_truth.SetMinimum(0)
        h_truth.SetMaximum(np.max(seg_mask))
        h_truth.Draw("COL")
        c_truth.Update()
        c_truth.SaveAs(os.path.join(output_dir, f"truth_{r}_{sr}_{evnum}_plane_{plane_name}.png"))

    def visualise_prediction(self, predictions, plane, r, sr, evnum, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        pred_labels = torch.argmax(predictions, dim=1).squeeze(0).cpu().numpy()  # Shape: (512, 512)
        plane_name = self.vis_config.PLANE_NAMES[plane]
        title = f"Plane {plane_name} Prediction (Run {r}, Subrun {sr}, Event {evnum})"

        c_pred = ROOT.TCanvas(f"c_pred_{plane}", title, 1200, 1200)
        c_pred.SetFillColor(ROOT.kWhite)
        c_pred.SetMargin(0.08, 0.08, 0.08, 0.08)

        h_pred = ROOT.TH2F(f"h_pred_{plane}", title, self.width, 0, self.width, self.height, 0, self.height)
        for i in range(self.width):
            for j in range(self.height):
                h_pred.SetBinContent(i + 1, j + 1, pred_labels[j, i])

        h_pred.GetXaxis().SetTitle("Local Drift Time")
        h_pred.GetYaxis().SetTitle("Local Wire Coord")
        h_pred.GetXaxis().SetTitleOffset(1.0)
        h_pred.GetYaxis().SetTitleOffset(1.0)
        h_pred.GetXaxis().SetLabelSize(0.03)
        h_pred.GetYaxis().SetLabelSize(0.03)
        h_pred.SetMinimum(0)
        h_pred.SetMaximum(np.max(pred_labels))
        h_pred.Draw("COL")
        c_pred.Update()
        c_pred.SaveAs(os.path.join(output_dir, f"pred_{r}_{sr}_{evnum}_plane_{plane_name}.png"))

def get_parser():
    parser = argparse.ArgumentParser(description="Inference and Visualisation Script")
    parser.add_argument("--model-path", type=str, default='/gluster/data/dune/niclane/checkpoints/segmentation/saved_models/unet_epoch_new_6.pt')
    parser.add_argument("--root-file", type=str, default="/gluster/data/dune/niclane/signal/nlane_prod_strange_resample_fhc_run2_fhc_reco2_reco2_trainingimage_signal_lambdamuon_1000_ana.root")
    parser.add_argument("--img-size", default=512, type=int)
    parser.add_argument("--output-dir", type=str, default="./displays")
    parser.add_argument("--target-labels", type=str, default="0,1,2,4")
    parser.add_argument("--plane", type=int, choices=[0, 1, 2], required=True)
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()

    foreground_labels = [int(x) for x in args.target_labels.split(',') if int(x) >= 2]
    dataset = ImageDataset(args, args.root_file, foreground_labels)

    vis_config = VisualisationConfig()
    visualiser = Visualiser(
        num_classes=len(foreground_labels),
        width=args.img_size,
        height=args.img_size,
        foreground_labels=foreground_labels,
        vis_config=vis_config
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.jit.load(args.model_path, map_location=device)
    model.eval()

    event_idx = visualiser._random_event_index(dataset)
    input_data, truth_data, r, sr, evnum = dataset.get_full_event(event_idx)
    input_tensor = torch.from_numpy(input_data).float().unsqueeze(0)  
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        predictions = model(input_tensor)  
        print(predictions)
        class_predictions = torch.argmax(predictions, dim=1)
        print(class_predictions)

    visualiser.visualise_truth(truth_data, args.plane, r, sr, evnum, args.output_dir)
    visualiser.visualise_prediction(predictions, args.plane, r, sr, evnum, args.output_dir)
    print(f"Event displays saved to {args.output_dir}")

if __name__ == "__main__":
    main()
