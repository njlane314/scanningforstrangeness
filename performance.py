import argparse
import os
import numpy as np
import ROOT

def get_parser():
    parser = argparse.ArgumentParser(description="Plotting script for segmentation training metrics using ROOT")
    parser.add_argument("--loss-file", type=str, default="/gluster/data/dune/niclane/checkpoints/segmentation/losses_20250314_135526.npz",
                        help="Path to the .npz file containing loss and metric results")
    parser.add_argument("--output-dir", type=str, default="./plots",
                        help="Directory to save the generated plots")
    return parser

def plot_loss(train_loss, valid_loss, output_dir):
    n_points = len(train_loss)
    x = np.arange(n_points, dtype=np.float64)
    train_graph = ROOT.TGraph(n_points, x, train_loss.astype(np.float64))
    valid_graph = ROOT.TGraph(n_points, x, valid_loss.astype(np.float64))
    train_graph.SetMarkerColor(ROOT.kBlue)
    train_graph.SetMarkerStyle(20)
    train_graph.SetMarkerSize(0.8)
    valid_graph.SetMarkerColor(ROOT.kRed)
    valid_graph.SetMarkerStyle(20)
    valid_graph.SetMarkerSize(0.8)
    canvas = ROOT.TCanvas("canvas", "", 800, 600)
    canvas.SetMargin(0.15, 0.05, 0.15, 0.05)  # Increased left/bottom margins
    train_graph.Draw("AP")  # P for points only
    valid_graph.Draw("P SAME")
    legend = ROOT.TLegend(0.75, 0.85, 0.95, 0.95)
    legend.SetBorderSize(0)
    legend.SetTextSize(0.03)
    legend.AddEntry(train_graph, "Training Loss", "p")
    legend.AddEntry(valid_graph, "Validation Loss", "p")
    legend.Draw()
    canvas.SaveAs(os.path.join(output_dir, "loss_plot.pdf"))

def plot_learning_rate(learning_rate, output_dir):
    n_points = len(learning_rate)
    x = np.arange(n_points, dtype=np.float64)
    lr_graph = ROOT.TGraph(n_points, x, learning_rate.astype(np.float64))
    lr_graph.SetMarkerColor(ROOT.kGreen + 2)
    lr_graph.SetMarkerStyle(20)
    lr_graph.SetMarkerSize(0.8)
    canvas = ROOT.TCanvas("canvas", "", 800, 600)
    canvas.SetMargin(0.15, 0.05, 0.15, 0.05)
    lr_graph.Draw("AP")
    legend = ROOT.TLegend(0.75, 0.85, 0.95, 0.95)
    legend.SetBorderSize(0)
    legend.SetTextSize(0.03)
    legend.AddEntry(lr_graph, "Learning Rate", "p")
    legend.Draw()
    canvas.SaveAs(os.path.join(output_dir, "learning_rate_plot.pdf"))

def plot_metrics(metric_data, metric_name, num_planes, output_dir):
    n_points = metric_data.shape[0]
    x = np.arange(n_points, dtype=np.float64)
    graphs = []
    colors = [ROOT.kBlue, ROOT.kRed, ROOT.kGreen + 2, ROOT.kMagenta]

    for plane in range(num_planes):
        plane_data = metric_data[:, plane].astype(np.float64)
        graph = ROOT.TGraph(n_points, x, plane_data)
        graph.SetMarkerColor(colors[plane % len(colors)])
        graph.SetMarkerStyle(20)
        graph.SetMarkerSize(0.8)
        graphs.append(graph)
    canvas = ROOT.TCanvas("canvas", "", 800, 600)
    canvas.SetMargin(0.15, 0.05, 0.15, 0.05)
    for i, graph in enumerate(graphs):
        draw_option = "AP" if i == 0 else "P SAME"
        graph.Draw(draw_option)
    legend = ROOT.TLegend(0.75, 0.85, 0.95, 0.95)
    legend.SetBorderSize(0)
    legend.SetTextSize(0.03)
    for plane, graph in enumerate(graphs):
        legend.AddEntry(graph, f"Plane {plane}", "p")
    legend.Draw()
    canvas.SaveAs(os.path.join(output_dir, f"{metric_name.lower()}_plot.pdf"))

def main():
    parser = get_parser()
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    try:
        data = np.load(args.loss_file)
    except FileNotFoundError:
        print(f"Error: Could not find file '{args.loss_file}'")
        return
    except Exception as e:
        print(f"Error loading file '{args.loss_file}': {e}")
        return

    train_loss = data['train_loss']
    valid_loss = data['valid_loss']
    learning_rate = data['learning_rate']
    jaccard = data['jaccard']
    dice = data['dice']
    accuracy = data['accuracy']
    recall = data['recall']

    num_planes = jaccard.shape[1] if jaccard.ndim > 1 else 1

    plot_loss(train_loss, valid_loss, args.output_dir)
    plot_learning_rate(learning_rate, args.output_dir)
    plot_metrics(jaccard, "Jaccard", num_planes, args.output_dir)
    plot_metrics(dice, "Dice", num_planes, args.output_dir)
    plot_metrics(accuracy, "Accuracy", num_planes, args.output_dir)
    plot_metrics(recall, "Recall", num_planes, args.output_dir)

    print(f"Plots saved to {args.output_dir}")

if __name__ == "__main__":
    ROOT.gROOT.SetBatch(True)
    main()