import argparse
import os
import uproot
import numpy as np
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
        self.img_size = args.img_size
        self.foreground_labels = foreground_labels
        self.num_classes = len(foreground_labels)
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
            entry_start=actual_idx, entry_stop=actual_idx + 1,
            library="np"
        )
        input_data = data["input"][0]  
        truth_data = data["truth"][0]  
        run = data["run"][0]
        subrun = data["subrun"][0]
        event = data["event"][0]
        return input_data, truth_data, run, subrun, event

@dataclass
class VisualizationConfig:
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
        self.vis_config.LEGEND_LABELS = {
            1: "Noise",
            2: "Muon",
            3: "Charged Kaon",
            4: "Kaon Short",
            5: "Lambda",
            6: "Charged Sigma"
        }  

    def _random_event_index(self, dataset) -> int:
        return np.random.randint(0, len(dataset))

    def visualise_input(self, input_data, r, sr, evnum, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        for plane in range(3):
            plane_iterable = input_data[plane]
            plane_array = np.fromiter(plane_iterable, dtype=np.float32, count=self.width*self.height) 
            input_img = plane_array.reshape(self.height, self.width)
            threshold = 10
            min_value = 10
            input_img = np.where(input_img > threshold, input_img, min_value)
            data_max = np.max(input_img)
            plane_name = self.vis_config.PLANE_NAMES[plane]
            title = f"Plane {plane_name} Input (Run {r}, Subrun {sr}, Event {evnum})"
            
            c_input = ROOT.TCanvas(f"c_input_{plane}", title, 1200, 1200)
            c_input.SetFillColor(ROOT.kWhite)
            c_input.SetLeftMargin(0.08)
            c_input.SetRightMargin(0.08)
            c_input.SetBottomMargin(0.08)
            c_input.SetTopMargin(0.08)
            
            h_input = ROOT.TH2F(f"h_input_{plane}", title, self.width, 0, self.width, self.height, 0, self.height)
            h_input.Reset()
            for i in range(self.width):
                for j in range(self.height):
                    h_input.SetBinContent(i + 1, j + 1, input_img[j, i])
            
            h_input.GetXaxis().SetTitle("Local Drift Time")
            h_input.GetYaxis().SetTitle("Local Wire Coord")
            h_input.GetXaxis().SetTitleOffset(1.0)
            h_input.GetYaxis().SetTitleOffset(1.0)
            h_input.GetXaxis().SetLabelSize(0.03)
            h_input.GetYaxis().SetLabelSize(0.03)
            h_input.GetXaxis().SetTitleSize(0.03)
            h_input.GetYaxis().SetTitleSize(0.03)
            h_input.GetXaxis().SetLabelColor(ROOT.kBlack)
            h_input.GetYaxis().SetLabelColor(ROOT.kBlack)
            h_input.GetXaxis().SetTitleColor(ROOT.kBlack)
            h_input.GetYaxis().SetTitleColor(ROOT.kBlack)
            h_input.GetXaxis().SetNdivisions(1)
            h_input.GetYaxis().SetNdivisions(1)
            h_input.GetXaxis().SetTickLength(0)
            h_input.GetYaxis().SetTickLength(0)
            h_input.GetXaxis().CenterTitle()
            h_input.GetYaxis().CenterTitle()
            h_input.SetMinimum(min_value)
            h_input.SetMaximum(data_max)
            
            h_input.Draw("COL")
            c_input.SetLogz(1)  
            c_input.Update()
            c_input.SaveAs(os.path.join(output_dir, f"input_{r}_{sr}_{evnum}_plane_{plane_name}.png"))

    def visualise_truth(self, truth_data, r, sr, evnum, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        num_planes = 3

        truth_data_np = np.array([
            np.fromiter(vec, dtype=np.float32, count=self.height * self.width).reshape(self.height, self.width)
            for vec in truth_data
        ])
        assert truth_data_np.shape == (3, 512, 512), f"Expected shape (3, 512, 512), got {truth_data_np.shape}"

        for plane in range(num_planes):
            seg_mask = truth_data_np[plane]
            unique_labels = np.unique(seg_mask)
            print(f"Plane {plane} unique labels: {unique_labels}")
            if not np.all(np.mod(seg_mask, 1) == 0):
                print(f"Warning: Non-integer values in truth data for plane {plane}, converting to int")
            seg_mask = seg_mask.astype(np.int64)

            plane_name = self.vis_config.PLANE_NAMES[plane]
            title = f"Plane {plane_name} Truth (Run {r}, Subrun {sr}, Event {evnum})"

            c_truth = ROOT.TCanvas(f"c_truth_{plane}", title, 1200, 1200)
            c_truth.SetFillColor(ROOT.kWhite)
            c_truth.SetLeftMargin(0.08)
            c_truth.SetRightMargin(0.08)
            c_truth.SetBottomMargin(0.08)
            c_truth.SetTopMargin(0.08)

            h_truth = ROOT.TH2F(f"h_truth_{plane}", title, self.width, 0, self.width, self.height, 0, self.height)
            h_truth.Reset()
            for i in range(self.width):
                for j in range(self.height):
                    h_truth.SetBinContent(i + 1, j + 1, seg_mask[j, i])

            h_truth.GetXaxis().SetTitle("Local Drift Time")
            h_truth.GetYaxis().SetTitle("Local Wire Coord")
            h_truth.GetXaxis().SetTitleOffset(1.0)
            h_truth.GetYaxis().SetTitleOffset(1.0)
            h_truth.GetXaxis().SetLabelSize(0.03)
            h_truth.GetYaxis().SetLabelSize(0.03)
            h_truth.GetXaxis().SetTitleSize(0.03)
            h_truth.GetYaxis().SetTitleSize(0.03)
            h_truth.GetXaxis().SetLabelColor(ROOT.kBlack)
            h_truth.GetYaxis().SetLabelColor(ROOT.kBlack)
            h_truth.GetXaxis().SetTitleColor(ROOT.kBlack)
            h_truth.GetYaxis().SetTitleColor(ROOT.kBlack)
            h_truth.GetXaxis().SetNdivisions(1)
            h_truth.GetYaxis().SetNdivisions(1)
            h_truth.GetXaxis().SetTickLength(0)
            h_truth.GetYaxis().SetTickLength(0)
            h_truth.GetXaxis().CenterTitle()
            h_truth.GetYaxis().CenterTitle()
            h_truth.SetMinimum(0)  
            h_truth.SetMaximum(np.max(seg_mask))  

            h_truth.Draw("COL")
            c_truth.Update()
            c_truth.SaveAs(os.path.join(output_dir, f"truth_{r}_{sr}_{evnum}_plane_{plane_name}.png"))

def get_parser():
    parser = argparse.ArgumentParser(description="Inference and Visualisation Script")
    parser.add_argument("--model-path", type=str, default="/gluster/data/dune/niclane/checkpoints/segmentation/uresnet_plane0_9_20250321_135456_ts.pt")
    parser.add_argument("--labels-path", type=str, default="/gluster/data/dune/niclane/checkpoints/segmentation/label_mapping_plane0.npz")
    parser.add_argument("--root-file", type=str, default="/gluster/data/dune/niclane/signal/nlane_prod_strange_resample_fhc_run2_fhc_reco2_reco2_trainingimage_signal_lambdamuon_1000_ana.root")
    parser.add_argument("--img-size", default=512, type=int)
    parser.add_argument("--output-dir", type=str, default="./displays")
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()
    label_mapping = np.load(args.labels_path)
    foreground_labels = [int(k) for k in label_mapping.keys()]
    
    dataset = ImageDataset(args, args.root_file, foreground_labels)
    vis_config = VisualizationConfig()
    visualiser = Visualiser(
        num_classes=len(foreground_labels),
        width=args.img_size,
        height=args.img_size,
        foreground_labels=foreground_labels,
        vis_config=vis_config
    )
    
    event_idx = visualiser._random_event_index(dataset)
    input_data, truth_data, r, sr, evnum = dataset.get_full_event(event_idx)
    visualiser.visualise_input(input_data, r, sr, evnum, args.output_dir)
    visualiser.visualise_truth(truth_data, r, sr, evnum, args.output_dir)
    
    print(f"Event displays saved to {args.output_dir}")

if __name__ == "__main__":
    main()