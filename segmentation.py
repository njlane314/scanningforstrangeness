import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
import uproot
import awkward as ak
import numpy as np
import os
import math
import random
from torch.optim.lr_scheduler import LambdaLR
from torch.cuda.amp import GradScaler
import contextlib

ROOT_FILE_PATH = "filtered_data.root" 
TREE_NAME = "numi_fhc_overlay_inclusive_genie_run1_run1" 
MODEL_OUTPUT_DIR = "trained_unet_segmentation_common_loader"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_DATALOADER_WORKERS = 0
SEED = 42

RAW_IMAGE_BRANCH_NAMES = ["raw_image_u", "raw_image_v", "raw_image_w"]
RECO_IMAGE_BRANCH_NAMES = ["reco_image_u", "reco_image_v", "reco_image_w"] 
TRUTH_IMAGE_BRANCH_NAMES = ["truth_image_u", "truth_image_v", "truth_image_w"]
EVENT_CATEGORY_BRANCH_NAME = "event_category"




# --- Segmentation Specific Constants ---
IMAGE_HEIGHT = 512 
IMAGE_WIDTH = 512  
TARGET_PLANE_IDX = 0 # 0 for u, 1 for v, 2 for w 

SEGMENTATION_NUM_EPOCHS = 10
SEGMENTATION_WARMUP_EPOCHS = 2
SEGMENTATION_BATCH_SIZE = 16
SEGMENTATION_LEARNING_RATE = 0.001
AMP_ENABLED = True

UNET_INPUT_CHANNELS = 2 # raw + reco
UNET_DEPTH = 4
UNET_N_FILTERS = 16
UNET_DROP_PROB = 0.1

# SEGMENTATION_TARGET_CATEGORIES = [0, 1, 2] 
# SIGNAL_CATEGORIES_TO_EXCLUDE_ENTIRELY = [10, 11] 

os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
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
        if layer.bias is not None:
            layer.bias.data.zero_()

class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out, k_size=3, k_pad=1):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, kernel_size=k_size, padding=k_pad, stride=1)
        self.norm1 = nn.GroupNorm(8, c_out)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(c_out, c_out, kernel_size=k_size, padding=k_pad, stride=1)
        self.norm2 = nn.GroupNorm(8, c_out)
        self.identity = nn.Conv2d(c_in, c_out, kernel_size=1, padding=0, stride=1) if c_in != c_out else nn.Identity()
        reinit_layer(self.conv1)
        reinit_layer(self.conv2)
        if isinstance(self.identity, nn.Conv2d):
             reinit_layer(self.identity)

    def forward(self, x):
        identity = self.identity(x)
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        return self.relu(out + identity)

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
    def __init__(self, in_dim, n_classes, depth, n_filters, drop_prob):
        super(UNet, self).__init__()
        self.ds_convs = nn.ModuleList()
        self.ds_maxpools = nn.ModuleList()
        self.ds_dropouts = nn.ModuleList()

        current_channels = in_dim
        for i in range(depth):
            out_channels = n_filters * (2**i)
            self.ds_convs.append(ConvBlock(current_channels, out_channels))
            self.ds_maxpools.append(maxpool())
            self.ds_dropouts.append(dropout(drop_prob))
            current_channels = out_channels

        self.bridge = ConvBlock(current_channels, current_channels * 2)
        current_channels *= 2

        self.us_tconvs = nn.ModuleList()
        self.us_convs = nn.ModuleList()
        self.us_dropouts = nn.ModuleList()

        for i in reversed(range(depth)):
            skip_channels = n_filters * (2**i)
            transpose_out_channels = skip_channels
            self.us_tconvs.append(TransposeConvBlock(current_channels, transpose_out_channels))
            self.us_dropouts.append(dropout(drop_prob))
            self.us_convs.append(ConvBlock(transpose_out_channels + skip_channels, transpose_out_channels))
            current_channels = transpose_out_channels

        self.output = nn.Conv2d(current_channels, n_classes, kernel_size=1)
        reinit_layer(self.output)

    def forward(self, x):
        skip_connections = []
        res = x

        for i in range(len(self.ds_convs)):
            res = self.ds_convs[i](res)
            skip_connections.append(res)
            res = self.ds_maxpools[i](res)
            res = self.ds_dropouts[i](res)

        res = self.bridge(res)
        skip_connections = skip_connections[::-1]

        for i in range(len(self.us_tconvs)):
            res = self.us_tconvs[i](res)
            if res.shape[2:] != skip_connections[i].shape[2:]:
                target_size = skip_connections[i].shape[2:]
                res = torch.nn.functional.interpolate(res, size=target_size, mode='bilinear', align_corners=False)
            res = torch.cat([res, skip_connections[i]], dim=1)
            res = self.us_dropouts[i](res)
            res = self.us_convs[i](res)
        return self.output(res)

class RootSegmentationDataset(Dataset):
    def __init__(self, root_file_path, tree_name,
                 raw_image_branch_list, reco_image_branch_list, TRUTH_IMAGE_branch_list,
                 event_category_branch_name, 
                 target_plane_idx, image_height, image_width,
                 # segmentation_target_categories=None, 
                 # signal_categories_to_exclude=None,  
                 dataset_name="SegmentationDataset"):
        super().__init__()
        self.root_file_path = root_file_path
        self.tree_name = tree_name
        self.raw_image_branch_list = raw_image_branch_list
        self.reco_image_branch_list = reco_image_branch_list
        self.TRUTH_IMAGE_branch_list = TRUTH_IMAGE_branch_list 
        self.event_category_branch_name = event_category_branch_name 
        self.target_plane_idx = target_plane_idx
        self.image_height = image_height
        self.image_width = image_width
        # self.segmentation_target_categories = set(segmentation_target_categories) if segmentation_target_categories else set()
        # self.signal_categories_to_exclude = set(signal_categories_to_exclude) if signal_categories_to_exclude else set()
        self.dataset_name = dataset_name
        self.event_indices = [] 
        self.root_file_handle = None
        self.tree = None

        try:
            self._open_root_file()
            self._prepare_indices()
        except Exception as e:
            print(f"CRITICAL ERROR: {self.dataset_name}: Init Error: {e}", flush=True)
            self._close_root_file()
            raise

    def _open_root_file(self):
        if not self.root_file_handle:
            self.root_file_handle = uproot.open(self.root_file_path, num_workers=0) 
            self.tree = self.root_file_handle[self.tree_name]

    def _close_root_file(self):
        if self.root_file_handle:
            self.root_file_handle.close()
            self.root_file_handle = None
            self.tree = None

    def close(self):
        self._close_root_file()

    def _prepare_indices(self):
        print(f"INFO: {self.dataset_name}: Preparing event indices for plane {self.target_plane_idx} from {self.root_file_path} tree {self.tree_name}", flush=True)
        if not self.tree:
            print(f"ERROR: {self.dataset_name}: ROOT tree not available.", flush=True)
            return

        # Basic filtering: use all events.
        # event_categories_data = self.tree[self.event_category_branch_name].array(library="ak")
        # for event_idx, event_cat in enumerate(event_categories_data):
        #     is_target = not self.segmentation_target_categories or event_cat in self.segmentation_target_categories
        #     is_excluded = event_cat in self.signal_categories_to_exclude
        #     if is_target and not is_excluded:
        #         self.event_indices.append(event_idx)
        
        num_entries = self.tree.num_entries
        self.event_indices = list(range(num_entries)) 
        print(f"INFO: {self.dataset_name}: Prepared {len(self.event_indices)} event indices (using all events).", flush=True)


    def __len__(self):
        return len(self.event_indices)

    def __getitem__(self, idx):
        if not self.root_file_handle or not self.tree:
             self._open_root_file()

        actual_event_idx = self.event_indices[idx]
        
        raw_branch_name = self.raw_image_branch_list[self.target_plane_idx]
        reco_branch_name = self.reco_image_branch_list[self.target_plane_idx]
        label_branch_name = self.TRUTH_IMAGE_branch_list[self.target_plane_idx]

        try:
            event_data = self.tree.arrays(
                [raw_branch_name, reco_branch_name, label_branch_name],
                entry_start=actual_event_idx,
                entry_stop=actual_event_idx + 1,
                library="ak"
            )

            raw_data_np = ak.to_numpy(event_data[raw_branch_name][0])
            reco_data_np = ak.to_numpy(event_data[reco_branch_name][0])

            raw_img_tensor = process_single_plane_image_data(raw_data_np, self.image_height, self.image_width)
            reco_img_tensor = process_single_plane_image_data(reco_data_np, self.image_height, self.image_width)
            
            input_image_tensor = torch.cat((raw_img_tensor, reco_img_tensor), dim=0)

            label_data_np = ak.to_numpy(event_data[label_branch_name][0])
            target_len = self.image_height * self.image_width
            current_len = len(label_data_np)

            if current_len == target_len:
                processed_label_np = label_data_np
            elif current_len > target_len:
                processed_label_np = label_data_np[:target_len]
            else:
                padding_value = 0 
                padding_dtype = label_data_np.dtype if label_data_np.size > 0 else np.int64 
                padding = np.full(target_len - current_len, padding_value, dtype=padding_dtype)
                processed_label_np = np.concatenate((label_data_np, padding))
            
            truth_tensor = torch.tensor(processed_label_np, dtype=torch.long).reshape(self.image_height, self.image_width)
            
            return input_image_tensor, truth_tensor
        except Exception as e:
            print(f"ERROR: {self.dataset_name}: GetItem Error for item_idx {idx} (actual_event_idx {actual_event_idx}): {e}", flush=True)
            import traceback
            traceback.print_exc()
            return torch.empty(UNET_INPUT_CHANNELS, self.image_height, self.image_width), torch.empty(self.image_height, self.image_width, dtype=torch.long)


def get_segmentation_scheduler(optimizer, warmup_epochs, total_epochs, initial_lr):
    warmup_epochs = max(1, warmup_epochs)
    total_epochs = max(total_epochs, warmup_epochs)
    def lr_lambda(current_epoch):
        if current_epoch < warmup_epochs:
            return float(current_epoch + 1) / float(warmup_epochs)
        else:
            progress = float(current_epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
            return 0.5 * (1.0 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)

def calculate_class_frequencies_from_loader(dataloader, num_classes_hint=None):
    class_counts = {}
    print(f"INFO: Calculating class frequencies from training data loader...", flush=True)
    num_batches_to_sample = min(len(dataloader), 20)
    if num_batches_to_sample == 0:
        print("WARNING: Training loader empty for class frequency calculation.", flush=True)
        return {i: 1 for i in range(num_classes_hint)} if num_classes_hint else {}

    processed_samples = 0
    for i, (_, labels_batch) in enumerate(dataloader):
        if i >= num_batches_to_sample: break
        for label_idx in range(labels_batch.shape[0]):
            label = labels_batch[label_idx]
            unique, counts = torch.unique(label, return_counts=True)
            for cls, count in zip(unique, counts):
                cls_item = cls.item()
                class_counts[cls_item] = class_counts.get(cls_item, 0) + count.item()
            processed_samples += labels_batch.shape[0]
    print(f"INFO: Class frequencies calculated from {processed_samples} samples.", flush=True)
    return class_counts

def calculate_class_weights(class_counts, num_total_classes):
    if not class_counts:
        print("WARNING: Class counts empty. Returning uniform weights.", flush=True)
        return torch.ones(num_total_classes)
    total_pixels = sum(class_counts.values())
    weights = torch.zeros(num_total_classes)
    for cls, count in class_counts.items():
        if 0 <= cls < num_total_classes and count > 0:
             weights[cls] = total_pixels / (num_total_classes * count)
        elif cls >= num_total_classes:
            print(f"WARNING: Class index {cls} out of range {num_total_classes}. Ignoring for weights.", flush=True)
    if (weights == 0).any():
        print("WARNING: Some classes not found in sample for weight calculation. Assigning mean weight or 1.0.", flush=True)
        non_zero_weights = weights[weights > 0]
        mean_weight = non_zero_weights.mean() if len(non_zero_weights) > 0 else 1.0
        weights[weights == 0] = mean_weight
    return weights

def train_unet_segmentation(
    model, train_loader, val_loader, criterion, optimizer, scheduler,
    num_epochs, device, model_dir, unet_input_channels, image_size):

    print(f"INFO: Starting U-Net Segmentation Training. Total Epochs: {num_epochs}. AMP: {AMP_ENABLED}", flush=True)
    all_train_batch_losses = []
    all_val_batch_losses = []
    all_learning_rates = []
    scaler = GradScaler(enabled=(AMP_ENABLED and device.type == 'cuda'))

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_train_loss_sum = 0.0
        num_train_batches = 0
        val_loader_iter = iter(val_loader) if val_loader and len(val_loader) > 0 else None

        for batch_idx, (train_images, train_labels) in enumerate(train_loader):
            if train_images.nelement() == 0 or train_labels.nelement() == 0 :
                print(f"WARNING: Epoch {epoch} Batch {batch_idx+1}: Empty batch from Dataloader. Skipping.", flush=True)
                continue
            train_images, train_labels = train_images.to(device), train_labels.to(device)
            optimizer.zero_grad()
            autocast_context = torch.cuda.amp.autocast() if AMP_ENABLED and device.type == 'cuda' else contextlib.nullcontext()

            with autocast_context:
                train_outputs = model(train_images)
                train_loss = criterion(train_outputs, train_labels)
            current_train_loss_val = train_loss.item()
            all_train_batch_losses.append(current_train_loss_val)
            epoch_train_loss_sum += current_train_loss_val
            num_train_batches += 1

            current_val_loss_val = float('nan')
            if val_loader_iter:
                model.eval()
                with torch.no_grad(), autocast_context:
                    try: val_images, val_labels = next(val_loader_iter)
                    except StopIteration: val_loader_iter = iter(val_loader); val_images, val_labels = next(val_loader_iter, (None, None))
                    if val_images is not None and val_labels is not None and val_images.nelement() > 0 and val_labels.nelement() > 0:
                        val_images, val_labels = val_images.to(device), val_labels.to(device)
                        val_outputs = model(val_images)
                        val_loss = criterion(val_outputs, val_labels)
                        current_val_loss_val = val_loss.item()
                model.train()
            all_val_batch_losses.append(current_val_loss_val)
            all_learning_rates.append(optimizer.param_groups[0]['lr'])

            if (batch_idx + 1) % 20 == 0 or batch_idx == 0:
                print(f"INFO: Epoch [{epoch}/{num_epochs}] Batch [{batch_idx+1}/{len(train_loader)}] "
                      f"Loss: {current_train_loss_val:.4f} Val Loss: {current_val_loss_val:.4f} "
                      f"LR: {optimizer.param_groups[0]['lr']:.6e}", flush=True)

            if AMP_ENABLED and device.type == 'cuda':
                scaler.scale(train_loss).backward(); scaler.step(optimizer); scaler.update()
            else:
                train_loss.backward(); optimizer.step()

        avg_epoch_train_loss = epoch_train_loss_sum / num_train_batches if num_train_batches > 0 else 0
        print(f"INFO: Epoch [{epoch}/{num_epochs}] Average Training Loss: {avg_epoch_train_loss:.4f}", flush=True)
        if scheduler: scheduler.step()

        epoch_torchscript_path = os.path.join(model_dir, f"unet_epoch_{epoch}.pt")
        try:
            model.eval()
            example_input = torch.randn(1, unet_input_channels, image_size, image_size, device=device)
            traced_model_epoch = torch.jit.trace(model, example_input)
            traced_model_epoch.save(epoch_torchscript_path)
            print(f"INFO: Saved TorchScript model for epoch {epoch} to {epoch_torchscript_path}", flush=True)
        except Exception as e_trace:
            print(f"ERROR: Failed to save TorchScript model for epoch {epoch}: {e_trace}", flush=True)
        finally:
            model.train()

    final_torchscript_path = os.path.join(model_dir, "unet_segmentation_final.pt")
    try:
        model.eval()
        example_input = torch.randn(1, unet_input_channels, image_size, image_size, device=device)
        traced_model_final = torch.jit.trace(model, example_input)
        traced_model_final.save(final_torchscript_path)
        print(f"INFO: Final TorchScript model saved to {final_torchscript_path}", flush=True)
    except Exception as e_trace_final:
        print(f"ERROR: Failed to save Final TorchScript model: {e_trace_final}", flush=True)

    loss_file_path = os.path.join(model_dir, f"segmentation_batch_losses_epochs_1_to_{num_epochs}.npz")
    np.savez(loss_file_path,
             train_batch_losses=np.array(all_train_batch_losses),
             val_batch_losses=np.array(all_val_batch_losses),
             learning_rates=np.array(all_learning_rates))
    print(f"INFO: Train/Val batch losses and LRs saved to {loss_file_path}", flush=True)
    print("INFO: Training completed.", flush=True)

if __name__ == "__main__":
    print(f"INFO: Device: {DEVICE}. Output Dir: {MODEL_OUTPUT_DIR}", flush=True)
    print(f"INFO: Using ROOT File: {ROOT_FILE_PATH}, Tree: {TREE_NAME}", flush=True)
    print(f"INFO: Raw branches: {RAW_IMAGE_BRANCH_NAMES}, Reco branches: {RECO_IMAGE_BRANCH_NAMES}, Label branches: {TRUTH_IMAGE_BRANCH_NAMES}", flush=True)
    print(f"INFO: Config: Epochs={SEGMENTATION_NUM_EPOCHS}, Warmup={SEGMENTATION_WARMUP_EPOCHS}, Batch={SEGMENTATION_BATCH_SIZE}, LR={SEGMENTATION_LEARNING_RATE}", flush=True)
    print(f"INFO: ImageHeight={IMAGE_HEIGHT}, ImageWidth={IMAGE_WIDTH}, PlaneIndex={TARGET_PLANE_IDX}, UNET InChannels={UNET_INPUT_CHANNELS}", flush=True)

    full_dataset = RootSegmentationDataset(
        root_file_path=ROOT_FILE_PATH,
        tree_name=TREE_NAME,
        raw_image_branch_list=RAW_IMAGE_BRANCH_NAMES,
        reco_image_branch_list=RECO_IMAGE_BRANCH_NAMES,
        TRUTH_IMAGE_branch_list=TRUTH_IMAGE_BRANCH_NAMES,
        event_category_branch_name=EVENT_CATEGORY_BRANCH_NAME, 
        target_plane_idx=TARGET_PLANE_IDX,
        image_height=IMAGE_HEIGHT,
        image_width=IMAGE_WIDTH
        # segmentation_target_categories=None, 
        # signal_categories_to_exclude=None    
    )

    if len(full_dataset) == 0:
        print("CRITICAL ERROR: Dataset is empty. Exiting.", flush=True)
        if hasattr(full_dataset, 'close'): full_dataset.close()
        exit(1)

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    if train_size == 0 or val_size == 0:
        print(f"ERROR: Dataset too small to split. Train: {train_size}, Val: {val_size}. Exiting.", flush=True)
        if hasattr(full_dataset, 'close'): full_dataset.close()
        exit(1)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(SEED))
    print(f"INFO: Dataset split. Full: {len(full_dataset)}, Train: {len(train_dataset)}, Val: {len(val_dataset)}", flush=True)

    train_loader = DataLoader(train_dataset, batch_size=SEGMENTATION_BATCH_SIZE, shuffle=True, num_workers=NUM_DATALOADER_WORKERS, pin_memory=DEVICE.type=='cuda', drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=SEGMENTATION_BATCH_SIZE, shuffle=False, num_workers=NUM_DATALOADER_WORKERS, pin_memory=DEVICE.type=='cuda', drop_last=False)

    if len(train_loader) == 0:
        print("CRITICAL ERROR: Training DataLoader is empty. Exiting.", flush=True)
        if hasattr(full_dataset, 'close'): full_dataset.close()
        exit(1)

    temp_loader_for_stats = DataLoader(train_dataset, batch_size=SEGMENTATION_BATCH_SIZE, shuffle=False)
    max_class_id = -1; found_classes_set = set()
    num_inspect_batches = min(len(temp_loader_for_stats), 5)
    if num_inspect_batches == 0:
        print("CRITICAL ERROR: Cannot inspect batches to determine N_CLASSES.", flush=True); exit(1)
    for i, (_, labels_batch_inspect) in enumerate(temp_loader_for_stats):
        if labels_batch_inspect.nelement() == 0: continue 
        if i >= num_inspect_batches: break
        unique_in_batch = torch.unique(labels_batch_inspect)
        for ul_item in unique_in_batch: found_classes_set.add(ul_item.item())
        if len(unique_in_batch) > 0: max_class_id = max(max_class_id, unique_in_batch.max().item())

    N_CLASSES_DYNAMIC = max_class_id + 1 if max_class_id != -1 else 0
    if N_CLASSES_DYNAMIC == 0 and len(found_classes_set) > 0: N_CLASSES_DYNAMIC = max(found_classes_set) + 1
    if N_CLASSES_DYNAMIC == 0:
        print("CRITICAL ERROR: Could not determine N_CLASSES from data. Check label branches and content.", flush=True);
        if hasattr(full_dataset, 'close'): full_dataset.close()
        exit(1)
    print(f"INFO: Dynamically determined N_CLASSES = {N_CLASSES_DYNAMIC}. Found classes: {sorted(list(found_classes_set))}", flush=True)

    class_counts_map = calculate_class_frequencies_from_loader(temp_loader_for_stats, N_CLASSES_DYNAMIC)
    del temp_loader_for_stats

    if not class_counts_map and N_CLASSES_DYNAMIC > 0:
        class_weight_tensor = torch.ones(N_CLASSES_DYNAMIC).to(DEVICE)
        print("WARNING: Class counts calculation failed, using uniform weights.", flush=True)
    elif not class_counts_map and N_CLASSES_DYNAMIC == 0: 
        print("CRITICAL ERROR: N_CLASSES is 0 and class counts failed.", flush=True); exit(1)
    else:
        class_weight_tensor = calculate_class_weights(class_counts_map, N_CLASSES_DYNAMIC).to(DEVICE)
    print(f"INFO: Using class weights: {class_weight_tensor}", flush=True)

    print("INFO: Starting training from scratch.", flush=True)
    unet_model = UNet(in_dim=UNET_INPUT_CHANNELS, n_classes=N_CLASSES_DYNAMIC, depth=UNET_DEPTH, n_filters=UNET_N_FILTERS, drop_prob=UNET_DROP_PROB).to(DEVICE)
    adam_optimizer = optim.Adam(unet_model.parameters(), lr=SEGMENTATION_LEARNING_RATE, weight_decay=1e-5)
    lr_scheduler = get_segmentation_scheduler(adam_optimizer, SEGMENTATION_WARMUP_EPOCHS, SEGMENTATION_NUM_EPOCHS, SEGMENTATION_LEARNING_RATE)
    loss_criterion = nn.CrossEntropyLoss(weight=class_weight_tensor)

    try:
        train_unet_segmentation(
            unet_model, train_loader, val_loader, loss_criterion, adam_optimizer, lr_scheduler,
            SEGMENTATION_NUM_EPOCHS, DEVICE, MODEL_OUTPUT_DIR,
            UNET_INPUT_CHANNELS, IMAGE_HEIGHT 
        )
    except Exception as e_train:
        print(f"CRITICAL ERROR during training execution: {e_train}", flush=True)
        import traceback
        traceback.print_exc()
    finally:
        if hasattr(full_dataset, 'close'):
            full_dataset.close()
            print("INFO: Closed dataset file handle.", flush=True)
    print("--- Main Script Finished ---", flush=True)