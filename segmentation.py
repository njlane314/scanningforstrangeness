import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F
import uproot
import awkward as ak
import numpy as np
import random
import os
import math
from torch.optim.lr_scheduler import LambdaLR
import torchvision.models as models
from torch.cuda.amp import GradScaler
import contextlib
import segmentation_models_pytorch as smp

ROOT_FILE_PATH = "filtered_data.root"
TREE_NAME = "numi_fhc_overlay_inclusive_genie_run1_run1"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_DATALOADER_WORKERS = 0
SEED = 42

RAW_IMAGE_BRANCH_NAMES = ["raw_image_u", "raw_image_v", "raw_image_w"]
RECO_IMAGE_BRANCH_NAMES = ["reco_image_u", "reco_image_v", "reco_image_w"]
TRUTH_IMAGE_BRANCH_NAMES = ["truth_image_u", "truth_image_v", "truth_image_w"]

IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512

SEGMENTATION_INPUT_CHANNELS = 2
TRUTH_PADDING_VALUE = 255
SEGMENTATION_IGNORE_INDEX = TRUTH_PADDING_VALUE

SEGMENTATION_LEARNING_RATE = 1e-4
SEGMENTATION_BATCH_SIZE = 16
SEGMENTATION_NUM_EPOCHS = 50
WARMUP_EPOCHS = 5
SEGMENTATION_MODEL_OUTPUT_DIR = "trained_segmentation_models_v2"
SEGMENTATION_MODEL_FILENAME_FINAL = "smp_uresnet34_segmentation_combined_input_all_events_dynamic_classes_warmup.pth"
SEGMENTATION_MODEL_SAVE_PATH_FINAL = os.path.join(SEGMENTATION_MODEL_OUTPUT_DIR, SEGMENTATION_MODEL_FILENAME_FINAL)
SEGMENTATION_TORCHSCRIPT_FILENAME_FINAL = "smp_uresnet34_segmentation_combined_input_all_events_dynamic_classes_warmup.pt"
SEGMENTATION_TORCHSCRIPT_SAVE_PATH_FINAL = os.path.join(SEGMENTATION_MODEL_OUTPUT_DIR, SEGMENTATION_TORCHSCRIPT_FILENAME_FINAL)

NUM_CLASSES = -1

os.makedirs(SEGMENTATION_MODEL_OUTPUT_DIR, exist_ok=True)
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED); torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

def process_single_plane_image_data(plane_data_np, target_height, target_width):
    target_len = target_height * target_width; current_len = len(plane_data_np)
    if current_len == target_len: processed_plane = plane_data_np
    elif current_len > target_len: processed_plane = plane_data_np[:target_len]
    else: padding = np.zeros(target_len - current_len, dtype=plane_data_np.dtype); processed_plane = np.concatenate((plane_data_np, padding))
    min_val, max_val = processed_plane.min(), processed_plane.max()
    if max_val > min_val: processed_plane = (processed_plane - min_val) / (max_val - min_val)
    elif max_val == min_val and min_val != 0: processed_plane = np.ones_like(processed_plane)
    else: processed_plane = np.zeros_like(processed_plane)
    return torch.tensor(processed_plane, dtype=torch.float32).reshape(1, target_height, target_width)

def process_segmentation_truth_data(plane_data_np, target_height, target_width, pad_value=TRUTH_PADDING_VALUE):
    target_len = target_height * target_width
    current_len = len(plane_data_np)
    if current_len == target_len: processed_plane = plane_data_np
    elif current_len > target_len: processed_plane = plane_data_np[:target_len]
    else: padding = np.full(target_len - current_len, pad_value, dtype=plane_data_np.dtype); processed_plane = np.concatenate((plane_data_np, padding))
    return torch.tensor(processed_plane, dtype=torch.long).reshape(target_height, target_width)

class RootSegmentationDataset(Dataset):
    def __init__(self, root_file_path, tree_name,
                 raw_image_branch_names, reco_image_branch_names, truth_image_branch_names,
                 image_height, image_width,
                 truth_pad_value=TRUTH_PADDING_VALUE):
        super().__init__()
        self.root_file_path = root_file_path; self.tree_name = tree_name
        self.raw_image_branch_names = raw_image_branch_names; self.reco_image_branch_names = reco_image_branch_names
        self.truth_image_branch_names = truth_image_branch_names
        self.num_plane_types = len(raw_image_branch_names)
        self.image_height = image_height; self.image_width = image_width
        self.truth_pad_value = truth_pad_value
        self.event_plane_indices = []
        self.root_file_handle = None
        try:
            self._open_root_file()
            self._prepare_indices()
        except Exception as e:
            print(f"CRITICAL ERROR: SegmentationDataset Init Error: {e}", flush=True)
            self._close_root_file(); raise
    def _open_root_file(self):
        if not self.root_file_handle: self.root_file_handle = uproot.open(self.root_file_path); self.tree = self.root_file_handle[self.tree_name]
    def _close_root_file(self):
        if self.root_file_handle: self.root_file_handle.close(); self.root_file_handle = None
    def close(self): self._close_root_file()
    def _prepare_indices(self):
        print(f"INFO: SegmentationDataset: Preparing indices for all events and all planes.", flush=True)
        num_events = len(self.tree)
        for event_idx in range(num_events):
            for plane_idx in range(self.num_plane_types): self.event_plane_indices.append((event_idx, plane_idx))
        print(f"INFO: SegmentationDataset: Prepared {len(self.event_plane_indices)} event-plane items.", flush=True)
    def __len__(self): return len(self.event_plane_indices)
    def __getitem__(self, idx):
        if not self.root_file_handle: self._open_root_file()
        event_idx, plane_idx = self.event_plane_indices[idx]
        raw_branch = self.raw_image_branch_names[plane_idx]
        reco_branch = self.reco_image_branch_names[plane_idx]
        truth_branch = self.truth_image_branch_names[plane_idx]
        branches_to_fetch = [raw_branch, reco_branch, truth_branch]
        try:
            event_data = self.tree.arrays(branches_to_fetch, entry_start=event_idx, entry_stop=event_idx + 1, library="ak")
            raw_img_data = ak.to_numpy(event_data[raw_branch][0])
            reco_img_data = ak.to_numpy(event_data[reco_branch][0])
            truth_img_data = ak.to_numpy(event_data[truth_branch][0])
            processed_raw_plane = process_single_plane_image_data(raw_img_data, self.image_height, self.image_width)
            processed_reco_plane = process_single_plane_image_data(reco_img_data, self.image_height, self.image_width)
            input_tensor = torch.cat((processed_raw_plane, processed_reco_plane), dim=0)
            truth_tensor = process_segmentation_truth_data(truth_img_data, self.image_height, self.image_width, self.truth_pad_value)
            return input_tensor, truth_tensor
        except Exception as e:
            print(f"ERROR: SegmentationDataset: GetItem Error for item index {idx} (event_idx {event_idx}, plane_idx {plane_idx}): {e}", flush=True)
            return torch.empty(0, SEGMENTATION_INPUT_CHANNELS, self.image_height, self.image_width), torch.empty(0, self.image_height, self.image_width)

def calculate_num_classes(dataset_instance, max_items_to_scan=1000, pad_value=TRUTH_PADDING_VALUE):
    print(f"INFO: Calculating number of classes by scanning up to {max_items_to_scan} items...", flush=True)
    unique_labels = set()
    scan_len = min(len(dataset_instance), max_items_to_scan)
    if scan_len == 0: print("WARNING: Dataset is empty, cannot determine number of classes."); return None
    for i in range(scan_len):
        try:
            _, truth_tensor = dataset_instance[i]
            if truth_tensor.nelement() > 0:
                valid_labels = truth_tensor[truth_tensor != pad_value]
                unique_labels.update(valid_labels.unique().tolist())
        except Exception as e: print(f"Warning: Error fetching item {i} during class calculation: {e}", flush=True); continue
        if i % 100 == 0 and i > 0: print(f"Scanned {i}/{scan_len} items for class calculation...", flush=True)
    if not unique_labels: print("WARNING: No valid labels found in the scanned items. Cannot determine NUM_CLASSES."); return None
    min_label = min(unique_labels); max_label = max(unique_labels)
    print(f"INFO: Found unique labels: {sorted(list(unique_labels))}", flush=True)
    print(f"INFO: Min label: {min_label}, Max label: {max_label}", flush=True)
    if min_label < 0: print(f"WARNING: Minimum label is {min_label}, which is unusual. Check truth data.", flush=True)
    num_determined_classes = int(max_label) + 1
    print(f"INFO: Determined NUM_CLASSES = {num_determined_classes} (based on max_label + 1)", flush=True)
    return num_determined_classes

def get_cosine_schedule_with_warmup(optimizer, warmup_epochs, total_epochs, initial_lr):
    def lr_lambda(current_epoch):
        if current_epoch < warmup_epochs: return float(current_epoch + 1) / float(warmup_epochs)
        else:
            progress = float(current_epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
            return 0.5 * (1.0 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)

def train_segmentation_model(model, train_loader, val_loader, criterion, optimizer, scheduler,
                             epochs, device, model_dir, model_save_path_final, torchscript_save_path_final,
                             example_input_shape_for_trace, model_name_prefix="UNetSeg"):
    print(f"INFO: Starting training for {model_name_prefix}...", flush=True)
    all_train_batch_losses = []; all_val_epoch_losses = []
    amp_enabled = (device.type == 'cuda'); scaler = GradScaler(enabled=amp_enabled)
    best_val_loss = float('inf')
    for epoch in range(epochs):
        model.train(); epoch_train_losses_sum = 0.0; num_train_batches = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            if inputs.nelement() == 0 or targets.nelement() == 0 or inputs.size(0) == 0:
                print(f"WARNING: {model_name_prefix} Epoch {epoch+1} Train Batch {batch_idx+1}: Empty/invalid batch.", flush=True); continue
            inputs = inputs.to(device); targets = targets.to(device, dtype=torch.long)
            optimizer.zero_grad()
            autocast_context = torch.cuda.amp.autocast() if amp_enabled else contextlib.nullcontext()
            with autocast_context: outputs = model(inputs); loss = criterion(outputs, targets)
            current_train_loss_val = loss.item()
            all_train_batch_losses.append(current_train_loss_val)
            epoch_train_losses_sum += current_train_loss_val; num_train_batches += 1
            if amp_enabled: scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            else: loss.backward(); optimizer.step()
            if (batch_idx + 1) % 50 == 0: print(f"INFO: {model_name_prefix} E[{epoch+1}/{epochs}] B[{batch_idx+1}/{len(train_loader)}] L:{current_train_loss_val:.4f} LR:{optimizer.param_groups[0]['lr']:.6e}", flush=True)
        avg_epoch_train_loss = epoch_train_losses_sum / num_train_batches if num_train_batches > 0 else float('nan')
        print(f"INFO: {model_name_prefix} Epoch [{epoch+1}/{epochs}] Avg Training Loss: {avg_epoch_train_loss:.4f}", flush=True)
        avg_epoch_val_loss = float('nan')
        if val_loader and len(val_loader) > 0:
            model.eval(); epoch_val_loss_sum = 0.0; num_val_batches = 0
            with torch.no_grad():
                for inputs_val, targets_val in val_loader:
                    if inputs_val.nelement() == 0 or targets_val.nelement() == 0 or inputs_val.size(0) == 0: continue
                    inputs_val = inputs_val.to(device); targets_val = targets_val.to(device, dtype=torch.long)
                    with autocast_context: outputs_val = model(inputs_val); val_loss = criterion(outputs_val, targets_val)
                    epoch_val_loss_sum += val_loss.item(); num_val_batches += 1
            avg_epoch_val_loss = epoch_val_loss_sum / num_val_batches if num_val_batches > 0 else float('nan')
            all_val_epoch_losses.append(avg_epoch_val_loss)
            print(f"INFO: {model_name_prefix} Epoch [{epoch+1}/{epochs}] Avg Validation Loss: {avg_epoch_val_loss:.4f}", flush=True)
            if not np.isnan(avg_epoch_val_loss) and avg_epoch_val_loss < best_val_loss:
                best_val_loss = avg_epoch_val_loss
                torch.save(model.state_dict(), os.path.join(model_dir, f"{model_name_prefix}_best_val.pth"))
                print(f"INFO: Saved best validation model for epoch {epoch+1}", flush=True)
        else: all_val_epoch_losses.append(float('nan'))
        if scheduler: scheduler.step()
        epoch_state_dict_path = os.path.join(model_dir, f"{model_name_prefix}_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), epoch_state_dict_path)
    torch.save(model.state_dict(), model_save_path_final)
    print(f"INFO: {model_name_prefix} final state_dict saved to {model_save_path_final}", flush=True)
    try:
        model.eval(); example_input = torch.randn(1, *example_input_shape_for_trace, device=device)
        traced_model = torch.jit.trace(model, example_input); traced_model.save(torchscript_save_path_final)
        print(f"INFO: {model_name_prefix} final TorchScript model saved to {torchscript_save_path_final}", flush=True)
    except Exception as e: print(f"ERROR: Failed to save Final {model_name_prefix} TorchScript model: {e}", flush=True)
    batch_losses_save_path = os.path.join(model_dir, f"{model_name_prefix}_losses.npz")
    np.savez(batch_losses_save_path, train_batch_losses=np.array(all_train_batch_losses), val_epoch_losses=np.array(all_val_epoch_losses))
    print(f"INFO: {model_name_prefix} train/val losses saved to {batch_losses_save_path}", flush=True)

if __name__ == "__main__":
    print(f"INFO: Device: {DEVICE}. Target Img H,W: {IMAGE_HEIGHT},{IMAGE_WIDTH}.", flush=True)
    print(f"INFO: Segmentation Task: Input Channels: {SEGMENTATION_INPUT_CHANNELS}", flush=True)
    print(f"INFO: Truth image branches: {TRUTH_IMAGE_BRANCH_NAMES}", flush=True)
    print(f"INFO: Raw input image branches: {RAW_IMAGE_BRANCH_NAMES}", flush=True)
    print(f"INFO: Reco input image branches: {RECO_IMAGE_BRANCH_NAMES}", flush=True)
    print(f"INFO: Using combined raw+reco inputs for each plane type (u,v,w).")
    print(f"INFO: All event categories will be used for training.")
    print(f"\n--- Preparing Dataset & Calculating NUM_CLASSES ---", flush=True)
    temp_dataset_for_class_calc = RootSegmentationDataset(
        ROOT_FILE_PATH, TREE_NAME,
        RAW_IMAGE_BRANCH_NAMES, RECO_IMAGE_BRANCH_NAMES, TRUTH_IMAGE_BRANCH_NAMES,
        IMAGE_HEIGHT, IMAGE_WIDTH,
        truth_pad_value=TRUTH_PADDING_VALUE
    )
    if len(temp_dataset_for_class_calc) == 0: print("CRITICAL ERROR: Dataset empty. Cannot calculate NUM_CLASSES or train. Exiting.", flush=True); exit()
    NUM_CLASSES = calculate_num_classes(temp_dataset_for_class_calc, max_items_to_scan=2000, pad_value=TRUTH_PADDING_VALUE)
    if NUM_CLASSES is None or NUM_CLASSES <= 0: print(f"CRITICAL ERROR: Failed to determine valid NUM_CLASSES ({NUM_CLASSES}). Exiting.", flush=True); exit()
    print(f"INFO: Dynamically determined NUM_CLASSES = {NUM_CLASSES}", flush=True)
    full_segmentation_dataset = temp_dataset_for_class_calc
    print(f"\n--- Starting U-Net Segmentation Model Training ---", flush=True)
    try:
        if len(full_segmentation_dataset) > 0:
            train_size = int(0.8 * len(full_segmentation_dataset)); val_size = len(full_segmentation_dataset) - train_size
            seg_train_subset = None; seg_val_subset = None
            if train_size == 0 or val_size == 0:
                print(f"WARNING: Seg Dataset too small for split. Train:{train_size},Val:{val_size}. Using all for training.", flush=True)
                if len(full_segmentation_dataset) > 0 : seg_train_subset = full_segmentation_dataset
                else: raise ValueError("Dataset is empty, cannot proceed with training.")
            else:
                seg_train_subset, seg_val_subset = random_split(full_segmentation_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(SEED))
                print(f"INFO: Seg: Split full dataset of {len(full_segmentation_dataset)} into Train:{len(seg_train_subset)}, Val:{len(seg_val_subset)}", flush=True)
            seg_train_loader = DataLoader(seg_train_subset, batch_size=SEGMENTATION_BATCH_SIZE, shuffle=True, num_workers=NUM_DATALOADER_WORKERS, pin_memory=DEVICE.type=='cuda', drop_last=True)
            seg_val_loader = None
            if seg_val_subset and len(seg_val_subset) > 0: seg_val_loader = DataLoader(seg_val_subset, batch_size=SEGMENTATION_BATCH_SIZE, shuffle=False, num_workers=NUM_DATALOADER_WORKERS, pin_memory=DEVICE.type=='cuda', drop_last=False)
            if len(seg_train_loader) == 0: print(f"ERROR: Seg Train loader empty. Batches:{len(seg_train_loader)}. Skipping training.", flush=True)
            else:
                u_net_model = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=SEGMENTATION_INPUT_CHANNELS, classes=NUM_CLASSES).to(DEVICE)
                print(f"INFO: Using UResNet34 from segmentation-models-pytorch with {NUM_CLASSES} output classes.")
                criterion_seg = nn.CrossEntropyLoss(ignore_index=SEGMENTATION_IGNORE_INDEX)
                optimizer_seg = optim.Adam(u_net_model.parameters(), lr=SEGMENTATION_LEARNING_RATE)
                scheduler_seg = get_cosine_schedule_with_warmup(optimizer_seg, WARMUP_EPOCHS, SEGMENTATION_NUM_EPOCHS, initial_lr=SEGMENTATION_LEARNING_RATE)
                print(f"INFO: Using Cosine LR scheduler with {WARMUP_EPOCHS} warmup epochs.")
                example_trace_input_shape_seg = (SEGMENTATION_INPUT_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH)
                train_segmentation_model(
                    u_net_model, seg_train_loader, seg_val_loader,
                    criterion_seg, optimizer_seg, scheduler_seg,
                    SEGMENTATION_NUM_EPOCHS, DEVICE,
                    SEGMENTATION_MODEL_OUTPUT_DIR,
                    SEGMENTATION_MODEL_SAVE_PATH_FINAL,
                    SEGMENTATION_TORCHSCRIPT_SAVE_PATH_FINAL,
                    example_trace_input_shape_seg,
                    model_name_prefix="SMP_UResNet34_Seg_Combined"
                )
        else: print("WARNING: Segmentation dataset empty. Skipping segmentation model training.", flush=True)
    except Exception as e:
        print(f"CRITICAL ERROR: Segmentation Training Failed: {e}", flush=True)
        import traceback; traceback.print_exc()
    finally:
        if full_segmentation_dataset and hasattr(full_segmentation_dataset, 'close'): full_segmentation_dataset.close()
    print("\nScript finished.", flush=True)