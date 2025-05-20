import argparse
import torch
import torch.nn as nn
import uproot
import numpy as np
import os
import ROOT
from torch.utils.data import Dataset

ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetPalette(57)

def maxpool():
    return nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

def dropout(prob):
    return nn.Dropout(prob)

def reinit_layer(layer, leak=0.0, use_kaiming_normal=True):
    if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d)):
        if use_kaiming_normal:
            nn.init.kaiming_normal_(layer.weight, a=leak)
        else:
            nn.init.kaiming_uniform_(layer.weight, a=leak)
            layer.bias.data.zero_()

class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out, k_size=3, k_pad=1):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, kernel_size=k_size, padding=k_pad, stride=1)
        self.norm1 = nn.GroupNorm(8, c_out)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(c_out, c_out, kernel_size=k_size, padding=k_pad, stride=1)
        self.norm2 = nn.GroupNorm(8, c_out)
        self.identity = nn.Conv2d(c_in, c_out, kernel_size=1, padding=0, stride=1)
        reinit_layer(self.conv1)
        reinit_layer(self.conv2)
    def forward(self, x):
        identity = self.identity(x)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        return self.relu(x + identity)

class TransposeConvBlock(nn.Module):
    def __init__(self, c_in, c_out, k_size=3, k_pad=1):
        super(TransposeConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(c_in, c_out, kernel_size=k_size, padding=k_pad, output_padding=1, stride=2),
            nn.GroupNorm(8, c_out),
            nn.ReLU(inplace=True)
        )
        reinit_layer(self.block[0])
    def forward(self, x):
        return self.block(x)

class UNet(nn.Module):
    def __init__(self, in_dim, n_classes, depth=4, n_filters=16, drop_prob=0.1):
        super(UNet, self).__init__()
        self.ds_conv_1 = ConvBlock(in_dim, n_filters)
        self.ds_conv_2 = ConvBlock(n_filters, 2 * n_filters)
        self.ds_conv_3 = ConvBlock(2 * n_filters, 4 * n_filters)
        self.ds_conv_4 = ConvBlock(4 * n_filters, 8 * n_filters)
        self.ds_maxpool_1 = maxpool()
        self.ds_maxpool_2 = maxpool()
        self.ds_maxpool_3 = maxpool()
        self.ds_maxpool_4 = maxpool()
        self.ds_dropout_1 = dropout(drop_prob)
        self.ds_dropout_2 = dropout(drop_prob)
        self.ds_dropout_3 = dropout(drop_prob)
        self.ds_dropout_4 = dropout(drop_prob)
        self.bridge = ConvBlock(8 * n_filters, 16 * n_filters)
        self.us_tconv_4 = TransposeConvBlock(16 * n_filters, 8 * n_filters)
        self.us_tconv_3 = TransposeConvBlock(8 * n_filters, 4 * n_filters)
        self.us_tconv_2 = TransposeConvBlock(4 * n_filters, 2 * n_filters)
        self.us_tconv_1 = TransposeConvBlock(2 * n_filters, n_filters)
        self.us_conv_4 = ConvBlock(16 * n_filters, 8 * n_filters)
        self.us_conv_3 = ConvBlock(8 * n_filters, 4 * n_filters)
        self.us_conv_2 = ConvBlock(4 * n_filters, 2 * n_filters)
        self.us_conv_1 = ConvBlock(2 * n_filters, 1 * n_filters)
        self.us_dropout_4 = dropout(drop_prob)
        self.us_dropout_3 = dropout(drop_prob)
        self.us_dropout_2 = dropout(drop_prob)
        self.us_dropout_1 = dropout(drop_prob)
        self.output = nn.Conv2d(n_filters, n_classes, 1)
    def forward(self, x):
        res = x
        res = self.ds_conv_1(res); conv_stack_1 = res.clone()
        res = self.ds_maxpool_1(res)
        res = self.ds_dropout_1(res)
        res = self.ds_conv_2(res); conv_stack_2 = res.clone()
        res = self.ds_maxpool_2(res)
        res = self.ds_dropout_2(res)
        res = self.ds_conv_3(res); conv_stack_3 = res.clone()
        res = self.ds_maxpool_3(res)
        res = self.ds_dropout_3(res)
        res = self.ds_conv_4(res); conv_stack_4 = res.clone()
        res = self.ds_maxpool_4(res)
        res = self.ds_dropout_4(res)
        res = self.bridge(res)
        res = self.us_tconv_4(res)
        res = torch.cat([res, conv_stack_4], dim=1)
        res = self.us_dropout_4(res)
        res = self.us_conv_4(res)
        res = self.us_tconv_3(res)
        res = torch.cat([res, conv_stack_3], dim=1)
        res = self.us_dropout_3(res)
        res = self.us_conv_3(res)
        res = self.us_tconv_2(res)
        res = torch.cat([res, conv_stack_2], dim=1)
        res = self.us_dropout_2(res)
        res = self.us_conv_2(res)
        res = self.us_tconv_1(res)
        res = torch.cat([res, conv_stack_1], dim=1)
        res = self.us_dropout_1(res)
        res = self.us_conv_1(res)
        res = self.output(res)
        return res

class ImageDataset(Dataset):
    def __init__(self, args):
        self.args = args
        self.root_file = uproot.open(args.root_file, array_cache=None, num_workers=0)
        self.tree = self.root_file["strangenessFilter/EventSelectionFilter"]
        self.plane = args.plane
        self.img_size = args.img_size
        in_fiducial_data = self.tree["in_fiducial"].array(library="np")
        self.filtered_indices = np.where(in_fiducial_data == True)[0]
        plane_letters = ['u', 'v', 'w']
        self.calo_key = f"calo_pixels_{plane_letters[self.plane]}"
        self.reco_key = f"reco_pixels_{plane_letters[self.plane]}"
        self.label_key = f"label_pixels_{plane_letters[self.plane]}"
    def __len__(self):
        return len(self.filtered_indices)
    def __getitem__(self, idx):
        actual_idx = self.filtered_indices[idx]
        data = self.tree.arrays([self.calo_key, self.reco_key, self.label_key, "run", "sub", "evt"], 
                                entry_start=actual_idx, entry_stop=actual_idx + 1, library="np")
        img_calo = data[self.calo_key][0].reshape(self.img_size, self.img_size)
        img_reco = data[self.reco_key][0].reshape(self.img_size, self.img_size)
        img = np.stack([img_calo, img_reco], axis=0)
        img = torch.from_numpy(img).float()
        truth = data[self.label_key][0].reshape(self.img_size, self.img_size)
        truth = torch.from_numpy(truth).long()
        run = data["run"][0]
        sub = data["sub"][0]
        evt = data["evt"][0]
        return img, truth, run, sub, evt

def set_histogram_style(h):
    h.GetXaxis().SetTitle("Local Drift Time")
    h.GetYaxis().SetTitle("Local Wire Coordinate")
    h.GetXaxis().SetTitleOffset(1.0)
    h.GetYaxis().SetTitleOffset(1.0)
    h.GetXaxis().SetLabelSize(0.03)
    h.GetYaxis().SetLabelSize(0.03)
    h.GetXaxis().SetTitleSize(0.03)
    h.GetYaxis().SetTitleSize(0.03)
    h.GetXaxis().SetLabelColor(ROOT.kBlack)
    h.GetYaxis().SetLabelColor(ROOT.kBlack)
    h.GetXaxis().SetTitleColor(ROOT.kBlack)
    h.GetYaxis().SetTitleColor(ROOT.kBlack)
    h.GetXaxis().SetNdivisions(1)
    h.GetYaxis().SetNdivisions(1)
    h.GetXaxis().SetTickLength(0)
    h.GetYaxis().SetTickLength(0)
    h.GetXaxis().CenterTitle()
    h.GetYaxis().CenterTitle()
    h.SetStats(0)

def visualize_input(calo_img, plane_letter, run, sub, evt, output_path, img_size):
    ROOT.gStyle.SetOptStat(0)
    ROOT.gStyle.SetPalette(57)
    c = ROOT.TCanvas(f"c_input_{plane_letter}", "", 1200, 1200)
    c.SetFillColor(ROOT.kWhite)
    c.SetLeftMargin(0.08)
    c.SetRightMargin(0.08)
    c.SetBottomMargin(0.08)
    c.SetTopMargin(0.08)

    h = ROOT.TH2F(f"h_input_{plane_letter}", f"Plane {plane_letter} Input (Run {run}, Subrun {sub}, Event {evt})",
                  img_size, 0, img_size, img_size, 0, img_size)
    
    threshold = 4
    min_value = 4
    max_value = min_value
    for i in range(img_size):
        for j in range(img_size):
            value = calo_img[j, i]
            if value > threshold:
                h.SetBinContent(i + 1, j + 1, value)
                if value > max_value:
                    max_value = value
            else:
                h.SetBinContent(i + 1, j + 1, min_value)
    
    set_histogram_style(h)
    h.SetMinimum(min_value)
    h.SetMaximum(max_value)
    h.Draw("COL")
    c.SetLogz(1)
    c.Update()
    c.SaveAs(output_path)
    del c, h

def visualize_segmentation(seg_img, title_prefix, plane_letter, run, sub, evt, n_classes, output_path, color_indices):
    c = ROOT.TCanvas(f"c_{title_prefix.lower()}_{plane_letter}", "", 1200, 1200)
    c.SetFillColor(ROOT.kWhite)
    c.SetLeftMargin(0.08)
    c.SetRightMargin(0.08)
    c.SetBottomMargin(0.08)
    c.SetTopMargin(0.08)

    h = ROOT.TH2F(f"h_{title_prefix.lower()}_{plane_letter}", f"Plane {plane_letter} {title_prefix} (Run {run}, Subrun {sub}, Event {evt})",
                  seg_img.shape[1], 0, seg_img.shape[1], seg_img.shape[0], 0, seg_img.shape[0])
    
    for i in range(seg_img.shape[1]):
        for j in range(seg_img.shape[0]):
            h.SetBinContent(i + 1, j + 1, seg_img[j, i])
    
    set_histogram_style(h)
    palette = [color_indices[i % len(color_indices)] for i in range(n_classes)]
    palette_np = np.array(palette, dtype=np.int32)
    ROOT.gStyle.SetPalette(n_classes, palette_np)
    h.SetMinimum(-0.5)
    h.SetMaximum(n_classes - 0.5)
    h.SetContour(n_classes)
    for i in range(n_classes):
        h.SetContourLevel(i, i - 0.5)
    
    h.Draw("COL")
    c.Update()
    c.SaveAs(output_path)
    del c, h

def get_parser():
    parser = argparse.ArgumentParser(description="Inference script for U-Net model with ROOT visualization")
    parser.add_argument("--checkpoint-path", type=str, default="/gluster/data/dune/niclane/checkpoints/segmentation/unet_new_epoch_8.pth")
    parser.add_argument("--root-file", type=str, default="/gluster/data/dune/niclane/signal/nl_lambda_nohadrons_reco2_validation_2000_strangenessselectionfilter_1200_new_analysis.root")
    parser.add_argument("--plane", type=int, choices=[0, 1, 2], required=True)
    parser.add_argument("--img-size", type=int, default=512)
    parser.add_argument("--output-dir", type=str, default=".")
    parser.add_argument("--num-events", type=int, default=1, help="Number of events to process and visualize")
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    state_dict = torch.load(args.checkpoint_path, map_location=device)
    n_classes = 12#state_dict['output.weight'].shape[0]
    model = UNet(in_dim=2, n_classes=n_classes, depth=4, n_filters=16, drop_prob=0.1)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    dataset = ImageDataset(args)
    num_events = min(args.num_events, len(dataset))
    indices = np.random.choice(len(dataset), num_events, replace=False)
    plane_letter = ['U', 'V', 'W'][args.plane]

    truth_colors = [
        ROOT.kGray,
        ROOT.TColor.GetColor(0, 0, 255),
        ROOT.TColor.GetColor(255, 0, 0),
        ROOT.TColor.GetColor(0, 255, 0),
        ROOT.TColor.GetColor(255, 255, 0),
        ROOT.TColor.GetColor(255, 0, 255),
        ROOT.TColor.GetColor(0, 255, 255),
        ROOT.TColor.GetColor(255, 165, 0),
        ROOT.TColor.GetColor(128, 0, 128),
        ROOT.TColor.GetColor(0, 128, 128),
        ROOT.TColor.GetColor(128, 128, 0),
        ROOT.TColor.GetColor(128, 0, 0),
        ROOT.TColor.GetColor(0, 128, 0),
        ROOT.TColor.GetColor(0, 0, 128),
        ROOT.TColor.GetColor(128, 128, 128),
        ROOT.TColor.GetColor(255, 192, 203),
        ROOT.TColor.GetColor(255, 215, 0),
        ROOT.TColor.GetColor(75, 0, 130),
    ]
    
    for idx in indices:
        original_img, truth, run, sub, evt = dataset[idx]
        img = original_img.unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(img)
            subset_classes = [0, 1, 2, 10]
            all_classes = list(range(n_classes))
            other_classes = [cls for cls in all_classes if cls not in subset_classes]
            outputs[:, other_classes, :, :] = -1e9
            predicted = torch.argmax(outputs, dim=1).cpu().numpy()[0]
            calo_img = original_img[0].numpy()
            mask = calo_img > 0
            predicted[~mask] = 0
            truth = truth.numpy()
        
        input_path = os.path.join(args.output_dir, f"input_{run}_{sub}_{evt}_plane_{plane_letter}.png")
        truth_path = os.path.join(args.output_dir, f"truth_{run}_{sub}_{evt}_plane_{plane_letter}.png")
        predicted_path = os.path.join(args.output_dir, f"predicted_{run}_{sub}_{evt}_plane_{plane_letter}.png")
        
        visualize_input(calo_img, plane_letter, run, sub, evt, input_path, args.img_size)
        visualize_segmentation(truth, "Truth", plane_letter, run, sub, evt, n_classes, truth_path, truth_colors)
        visualize_segmentation(predicted, "Predicted", plane_letter, run, sub, evt, n_classes, predicted_path, truth_colors)
    
    print(f"Figures saved to {args.output_dir}")

if __name__ == "__main__":
    main()