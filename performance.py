import argparse
import os
import numpy as np
import ROOT

def get_parser():
    parser = argparse.ArgumentParser(description="Plotting script for segmentation training metrics using ROOT")
    parser.add_argument("--loss-file", type=str, default="/gluster/data/dune/niclane/checkpoints/segmentation/losses_new_epochs_1_to_4.npz",
                        help="Path to the .npz file containing loss and metric results")
    parser.add_argument("--output-dir", type=str, default="./plots",
                        help="Directory to save the generated plots")
    return parser

def plot_combined(train_loss, valid_loss, learning_rate, output_dir):
    n_points = len(train_loss)
    x = np.arange(n_points, dtype=np.float64)

    train_graph = ROOT.TGraph(n_points, x, train_loss.astype(np.float64))
    valid_graph = ROOT.TGraph(n_points, x, valid_loss.astype(np.float64))

    train_graph.SetLineColor(ROOT.kBlue)
    train_graph.SetLineWidth(1)
    train_graph.SetMarkerColor(ROOT.kBlue)
    train_graph.SetMarkerStyle(20)
    train_graph.SetMarkerSize(0.5)

    valid_graph.SetLineColor(ROOT.kRed)
    valid_graph.SetLineWidth(1)
    valid_graph.SetMarkerColor(ROOT.kRed)
    valid_graph.SetMarkerStyle(20)
    valid_graph.SetMarkerSize(0.5)

    loss_min = min(train_loss.min(), valid_loss.min())
    loss_max = max(train_loss.max(), valid_loss.max())
    if loss_min <= 0: 
        loss_min = 1e-10
    if loss_min == loss_max:  
        loss_max = loss_min * 10

    lr_min = learning_rate.min()
    lr_max = learning_rate.max()
    if lr_min <= 0: 
        lr_min = 1e-10
    if lr_min == lr_max: 
        lr_max = lr_min * 10

    print(f"Loss range: {loss_min} to {loss_max}")
    print(f"Learning rate range: {lr_min} to {lr_max}")

    lr_scaled = (learning_rate - lr_min) / (lr_max - lr_min) * (loss_max - loss_min) + loss_min
    lr_graph = ROOT.TGraph(n_points, x, lr_scaled.astype(np.float64))
    lr_graph.SetLineColor(ROOT.kGreen + 2)
    lr_graph.SetLineWidth(1)
    lr_graph.SetMarkerColor(ROOT.kGreen + 2)
    lr_graph.SetMarkerStyle(20)
    lr_graph.SetMarkerSize(0.5)

    canvas = ROOT.TCanvas("canvas", "", 800, 600)
    canvas.SetMargin(0.15, 0.15, 0.15, 0.05)

    mg = ROOT.TMultiGraph()
    mg.Add(train_graph)
    mg.Add(valid_graph)
    mg.SetTitle("")
    mg.GetXaxis().SetTitle("Iteration")
    mg.GetYaxis().SetTitle("Loss")
    mg.GetYaxis().SetRangeUser(loss_min, loss_max)

    canvas.SetLogy()
    mg.Draw("APL")

    lr_graph.Draw("PL SAME")

    axis = ROOT.TGaxis(mg.GetXaxis().GetXmax(), loss_min,
                       mg.GetXaxis().GetXmax(), loss_max,
                       lr_min, lr_max, 510, "+LG")
    axis.SetLineColor(ROOT.kGreen + 2)
    axis.SetLabelColor(ROOT.kGreen + 2)
    axis.SetTitle("Learning Rate")
    axis.SetTitleColor(ROOT.kGreen + 2)
    axis.SetTitleOffset(1.3)
    axis.SetLabelFont(42)
    axis.SetTitleFont(42)
    axis.SetMaxDigits(2)
    axis.Draw()

    legend = ROOT.TLegend(0.6, 0.75, 0.75, 0.9)
    legend.SetBorderSize(0)
    legend.SetFillStyle(0)
    legend.SetTextSize(0.035)
    legend.SetTextFont(42)
    legend.AddEntry(train_graph, "Training Loss", "lp")
    legend.AddEntry(valid_graph, "Validation Loss", "lp")
    legend.AddEntry(lr_graph, "Learning Rate", "lp")
    legend.Draw()

    canvas.SaveAs(os.path.join(output_dir, "loss_plot.pdf"))

def plot_metrics(metric_data, metric_name, num_planes, output_dir):
    n_points = metric_data.shape[0]
    x = np.arange(n_points, dtype=np.float64)
    graphs = []
    colors = [ROOT.kBlue, ROOT.kRed, ROOT.kGreen + 2, ROOT.kMagenta]

    for plane in range(num_planes):
        plane_data = metric_data[:, plane].astype(np.float64)
        graph = ROOT.TGraph(n_points, x, plane_data)
        graph.SetLineColor(colors[plane % len(colors)])
        graph.SetLineWidth(1)
        graph.SetMarkerColor(colors[plane % len(colors)])
        graph.SetMarkerStyle(20)
        graph.SetMarkerSize(0.5)
        graphs.append(graph)
    
    canvas = ROOT.TCanvas("canvas", "", 800, 600)
    canvas.SetMargin(0.15, 0.05, 0.15, 0.05)
    canvas.SetLogy()
    
    graphs[0].SetTitle("")
    graphs[0].GetXaxis().SetTitle("Epoch")
    graphs[0].GetYaxis().SetTitle(metric_name)
    
    for i, graph in enumerate(graphs):
        draw_option = "APL" if i == 0 else "PL SAME"
        graph.Draw(draw_option)
    
    legend = ROOT.TLegend(0.7, 0.8, 0.9, 0.9)
    legend.SetBorderSize(0)
    legend.SetFillStyle(0)
    legend.SetTextSize(0.035)
    legend.SetTextFont(42)
    for plane, graph in enumerate(graphs):
        legend.AddEntry(graph, f"Plane {plane}", "lp")
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

    train_loss = data['step_train_losses']
    valid_loss = data['step_valid_losses']
    learning_rate = data['step_learning_rates']

    plot_combined(train_loss, valid_loss, learning_rate, args.output_dir)

    print(f"Plots saved to {args.output_dir}")

if __name__ == "__main__":
    ROOT.gROOT.SetBatch(True)
    main()