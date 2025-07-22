import os
import numpy as np
import ROOT

LOSS_FILE_PATH = "training_cnn_metrics/mpid_metrics.npz"
OUTPUT_DIRECTORY = "plots_root_output_adapted"
CANVAS_TITLE = None
LEARNING_RATE_KEY = "batch_learning_rates"

def create_loss_plots_root(train_loss_data, valid_loss_data, learning_rate_data, output_directory, file_basename_for_output, plot_canvas_title_str):
    if not train_loss_data.size: 
        print(f"INFO: No training data points for {file_basename_for_output}. Plot generation skipped.", flush=True)
        return
    num_train_points = len(train_loss_data)
    x_coords_train = np.arange(num_train_points, dtype=np.float64)
    root_train_graph = ROOT.TGraph(num_train_points, x_coords_train, train_loss_data.astype(np.float64))
    root_train_graph.SetLineColor(ROOT.kBlue)
    root_train_graph.SetLineWidth(1)
    root_train_graph.SetMarkerColor(ROOT.kBlue)
    root_train_graph.SetMarkerStyle(20)
    root_train_graph.SetMarkerSize(0.5)

    valid_loss_indices = ~np.isnan(valid_loss_data)
    x_coords_valid = np.arange(len(valid_loss_data), dtype=np.float64)[valid_loss_indices]
    cleaned_valid_loss = valid_loss_data[valid_loss_indices]
    root_valid_graph = None
    if cleaned_valid_loss.size > 0:
        root_valid_graph = ROOT.TGraph(len(cleaned_valid_loss), x_coords_valid, cleaned_valid_loss.astype(np.float64))
        root_valid_graph.SetLineColor(ROOT.kRed)
        root_valid_graph.SetLineWidth(1)
        root_valid_graph.SetMarkerColor(ROOT.kRed)
        root_valid_graph.SetMarkerStyle(20)
        root_valid_graph.SetMarkerSize(0.5)
    else: 
        print(f"INFO: No actual validation loss points found for {file_basename_for_output}.", flush=True)

    min_loss_val_list = [np.nanmin(train_loss_data)] if train_loss_data.size > 0 else []
    max_loss_val_list = [np.nanmax(train_loss_data)] if train_loss_data.size > 0 else []
    if cleaned_valid_loss.size > 0: 
        min_loss_val_list.append(np.nanmin(cleaned_valid_loss))
        max_loss_val_list.append(np.nanmax(cleaned_valid_loss))

    actual_min_loss = np.nanmin(min_loss_val_list) if min_loss_val_list else 1e-10
    actual_max_loss = np.nanmax(max_loss_val_list) if max_loss_val_list else 1.0
    if actual_min_loss <= 0 or np.isnan(actual_min_loss): 
        actual_min_loss = 1e-10
    if np.isnan(actual_max_loss) or actual_max_loss <= actual_min_loss : 
        actual_max_loss = actual_min_loss * 10

    scaled_max_loss = actual_max_loss * 1.5

    root_lr_graph = None
    lr_axis_display = None
    has_lr_to_plot = False
    actual_lr_min = 0
    actual_lr_max = 0
    if learning_rate_data is not None and len(learning_rate_data) == num_train_points:
        lr_valid_indices_mask = ~np.isnan(learning_rate_data)
        lr_x_coords = x_coords_train[lr_valid_indices_mask]
        cleaned_lr_data = learning_rate_data[lr_valid_indices_mask]
        if cleaned_lr_data.size > 0:
            actual_lr_min = np.nanmin(cleaned_lr_data)
            actual_lr_max = np.nanmax(cleaned_lr_data)
            if actual_lr_min <= 0 or np.isnan(actual_lr_min): 
                actual_lr_min = 1e-10
            if np.isnan(actual_lr_max) or actual_lr_max <= actual_lr_min: 
                actual_lr_max = actual_lr_min * 10

            lr_scaled_values = np.full_like(cleaned_lr_data, np.nan, dtype=np.float64)
            range_lr = actual_lr_max - actual_lr_min
            range_loss = scaled_max_loss - actual_min_loss 
            if range_lr > 1e-12 and range_loss > 1e-12 : 
                lr_scaled_values = (cleaned_lr_data - actual_lr_min) / range_lr * range_loss + actual_min_loss
            else: 
                lr_scaled_values = np.full_like(cleaned_lr_data, actual_min_loss + range_loss * 0.5, dtype=np.float64)

            if lr_scaled_values.size > 0:
                root_lr_graph = ROOT.TGraph(len(lr_x_coords), lr_x_coords, lr_scaled_values.astype(np.float64))
                root_lr_graph.SetLineColor(ROOT.kGreen + 2)
                root_lr_graph.SetLineWidth(1)
                root_lr_graph.SetMarkerColor(ROOT.kGreen + 2)
                root_lr_graph.SetMarkerStyle(20)
                root_lr_graph.SetMarkerSize(0.5)
                has_lr_to_plot = True
        else: 
            print(f"INFO: No valid learning rate data points found for {file_basename_for_output}.", flush=True)

    plot_canvas = ROOT.TCanvas(f"canvas_{file_basename_for_output}", plot_canvas_title_str, 800, 600)
    plot_canvas.SetMargin(0.15, 0.15 if has_lr_to_plot else 0.05, 0.15, 0.05)
    plot_canvas.SetLogy()

    multi_graph_obj = ROOT.TMultiGraph()
    multi_graph_obj.Add(root_train_graph, "PL")
    if root_valid_graph: 
        multi_graph_obj.Add(root_valid_graph, "PL")
    multi_graph_obj.SetTitle(";Iteration;Loss")
    multi_graph_obj.Draw("APL")

    plot_canvas.Update() 
    
    if multi_graph_obj.GetYaxis():
        multi_graph_obj.GetYaxis().SetRangeUser(actual_min_loss, scaled_max_loss)
    if multi_graph_obj.GetHistogram():
        multi_graph_obj.GetHistogram().SetMinimum(actual_min_loss)
        multi_graph_obj.GetHistogram().SetMaximum(scaled_max_loss)
    
    plot_canvas.Update()

    if has_lr_to_plot and root_lr_graph:
        root_lr_graph.Draw("PL SAME")
        current_x_axis_max = multi_graph_obj.GetXaxis().GetXmax() if multi_graph_obj.GetXaxis() else num_train_points 
        
        lr_axis_display = ROOT.TGaxis(current_x_axis_max, actual_min_loss, current_x_axis_max, scaled_max_loss, actual_lr_min, actual_lr_max, 510, "+LG")
        lr_axis_display.SetLineColor(ROOT.kGreen + 2)
        lr_axis_display.SetLabelColor(ROOT.kGreen + 2)
        lr_axis_display.SetTitle("Learning Rate")
        lr_axis_display.SetTitleColor(ROOT.kGreen + 2)
        lr_axis_display.SetTitleOffset(1.3)
        lr_axis_display.SetLabelFont(42)
        lr_axis_display.SetTitleFont(42)
        lr_axis_display.SetMaxDigits(2)
        lr_axis_display.Draw()

    plot_legend = ROOT.TLegend(0.6, 0.75, 0.75, 0.9)
    plot_legend.SetBorderSize(0)
    plot_legend.SetFillStyle(0)
    plot_legend.SetTextSize(0.035)
    plot_legend.SetTextFont(42)
    plot_legend.AddEntry(root_train_graph, "Training Loss", "lp")
    if root_valid_graph: 
        plot_legend.AddEntry(root_valid_graph, "Validation Loss", "lp")
    if has_lr_to_plot and root_lr_graph: 
        plot_legend.AddEntry(root_lr_graph, "Learning Rate", "lp")
    plot_legend.Draw()

    plot_canvas.Update()
    plot_canvas.SaveAs(os.path.join(output_directory, f"{file_basename_for_output}_plot.pdf"))
    plot_canvas.SaveAs(os.path.join(output_directory, f"{file_basename_for_output}_plot.png"))
    print(f"INFO: Plot files saved in {output_directory} for {file_basename_for_output}", flush=True)

def main_executor():
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
    if not os.path.isfile(LOSS_FILE_PATH):
        print(f"ERROR: Specified loss file not found: {LOSS_FILE_PATH}", flush=True)
        return
    try:
        loss_data_npz = np.load(LOSS_FILE_PATH, allow_pickle=True)
    except Exception as e:
        print(f"ERROR: Failed to load loss file {LOSS_FILE_PATH}: {e}", flush=True)
        return

    train_losses_arr = loss_data_npz.get('train_batch_losses', np.array([]))
    
    if not train_losses_arr.size:
        print(f"ERROR: 'train_batch_losses' not found or is empty in {LOSS_FILE_PATH}.", flush=True)
        return
    
    num_train_batches_total = len(train_losses_arr)

    val_losses_arr = loss_data_npz.get('val_batch_losses', np.array([]))
    
    if not val_losses_arr.size:
        print(f"INFO: 'val_batch_losses' not found or is empty in {LOSS_FILE_PATH}. Validation loss will not be plotted.", flush=True)
        val_losses_arr = np.full(num_train_batches_total, np.nan)
    
    if val_losses_arr.size != train_losses_arr.size and train_losses_arr.size > 0:
        print(f"WARNING: Train ({train_losses_arr.size}) and Val ({val_losses_arr.size}) loss arrays have different sizes. Adjusting val_losses_arr to match train_losses_arr length by padding/truncating.", flush=True)
        temp_val = np.full_like(train_losses_arr, np.nan)
        common_len = min(len(val_losses_arr), len(temp_val))
        temp_val[:common_len] = val_losses_arr[:common_len]
        val_losses_arr = temp_val

    epoch_lr_data = loss_data_npz.get('epoch_learning_rates', np.array([]))
    
    lr_data_arr = np.full(num_train_batches_total, np.nan)
    if epoch_lr_data.size > 0 and num_train_batches_total > 0:
        num_epochs_lr = len(epoch_lr_data)
        if num_epochs_lr > 0:
            approx_batches_per_epoch_lr = num_train_batches_total // num_epochs_lr
            if approx_batches_per_epoch_lr == 0:
                approx_batches_per_epoch_lr = 1

            expanded_lr_list = []
            for lr_val in epoch_lr_data:
                expanded_lr_list.extend([lr_val] * approx_batches_per_epoch_lr)
            
            current_lr_len = len(expanded_lr_list)
            if current_lr_len < num_train_batches_total:
                expanded_lr_list.extend([epoch_lr_data[-1]] * (num_train_batches_total - current_lr_len))
            lr_data_arr = np.array(expanded_lr_list[:num_train_batches_total])
            print(f"INFO: Constructed batch-aligned learning rates from 'epoch_learning_rates'.", flush=True)
        else:
            print(f"INFO: No epoch_learning_rates found or cannot align with training batches.", flush=True)
    else:
        print(f"INFO: No 'epoch_learning_rates' found in {LOSS_FILE_PATH}. Learning rate will not be plotted.", flush=True)

    if np.all(np.isnan(lr_data_arr)) or lr_data_arr.size == 0 or lr_data_arr.size != train_losses_arr.size:
        print(f"INFO: Learning rate data could not be loaded or constructed to match training batches. LR will not be plotted.", flush=True)
        lr_data_arr = None

    input_file_basename = os.path.splitext(os.path.basename(LOSS_FILE_PATH))[0]
    current_plot_title = CANVAS_TITLE
    if current_plot_title is None:
        title_base = input_file_basename.replace('_metrics', '').replace('_final', '')
        title_base = title_base.replace('mpid', 'MPID Classifier')
        title_elements = title_base.replace("_", " ").split(' ')
        current_plot_title = ' '.join(word.capitalize() for word in title_elements if word) + " Performance"
        
    create_loss_plots_root(train_losses_arr, val_losses_arr, lr_data_arr, OUTPUT_DIRECTORY, input_file_basename, current_plot_title)
    print(f"INFO: Script execution finished for {LOSS_FILE_PATH}.", flush=True)

if __name__ == "__main__":
    ROOT.gROOT.SetBatch(True)
    main_executor()