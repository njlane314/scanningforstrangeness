import uproot
import numpy as np
import ROOT
import argparse

def get_parser():
    parser = argparse.ArgumentParser(description="Create histograms of label pixel percentages")
    parser.add_argument("--root-file", type=str, default="/gluster/data/dune/niclane/signal/nlane_prod_strange_resample_fhc_run2_fhc_reco2_reco2_trainingimage_signal_lambdamuon_1000_ana.root")
    parser.add_argument("--labels", type=str, default="0,1,2,4")
    parser.add_argument("--plane", type=int, default=0, choices=[0, 1, 2])
    parser.add_argument("--output-pdf", type=str, default="/gluster/home/niclane/scanningforstrangeness/plots/pixel_occupancy_histograms.pdf")
    parser.add_argument("--img-size", default=512, type=int)
    return parser

def main():
    args = get_parser().parse_args()
    labels = [int(x) for x in args.labels.split(',')]
    print(f"Analyzing labels: {labels}")
    plane = args.plane
    img_size = args.img_size
    total_pixels = img_size * img_size

    with uproot.open(args.root_file) as root_file:
        tree = root_file["imageanalyser/ImageTree"]
        event_types = tree["type"].array(library="np")
        indices = np.where(event_types == 0)[0]
        print(f"Processing {len(indices)} events where type == 0")

        min_percentage = 1e-6  # 10⁻⁶ %
        max_percentage = 100   # 100%
        n_bins = 50
        bin_edges = np.logspace(np.log10(min_percentage), np.log10(max_percentage), n_bins + 1)

        histos = {}
        sum_percentages = {label: 0.0 for label in labels}
        for label in labels:
            histos[label] = ROOT.TH1D(f"histo_label{label}_plane{plane}", 
                                      f"Label {label} pixel percentages, plane {plane}", 
                                      n_bins, bin_edges)
            histos[label].GetXaxis().SetTitle("Percentage of pixels / event (%)")
            histos[label].GetYaxis().SetTitle("Fraction of Dataset")

        for idx in indices:
            data = tree.arrays(["truth"], entry_start=idx, entry_stop=idx + 1, library="np")
            truth = data["truth"][0][plane]
            if idx == indices[0]:
                print(f"Unique values in truth, plane {plane}: {np.unique(truth)}")
            for label in labels:
                count = np.sum(truth == label)
                print(count)
                percentage = (count / total_pixels) * 100
                print(percentage)
                if percentage > 0:
                    histos[label].Fill(percentage)
                sum_percentages[label] += percentage

    canvas = ROOT.TCanvas("canvas", "Histograms", 800, 600)
    canvas.SetLogx(1)  
    canvas.Print(args.output_pdf + "[")  
    for label in labels:
        histo = histos[label]
        histo.Scale(1.0 / len(indices))  
        avg = sum_percentages[label] / len(indices)
        print(f"Average percentage for label {label}: {avg:.4f}%")
        histo.Draw()
        canvas.Print(args.output_pdf)  

    canvas.Print(args.output_pdf + "]")  
    print(f"Histograms saved to {args.output_pdf}")

if __name__ == "__main__":
    main()