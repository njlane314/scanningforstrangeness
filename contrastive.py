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
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
import torchvision.models as models
from torch.cuda.amp import GradScaler
import contextlib
from datetime import datetime, timezone 
import traceback 

# --- Global Configuration & Constants ---
ROOT_FILE_PATH = "filtered_data.root"
TREE_NAME = "numi_fhc_overlay_inclusive_genie_run1_run1"
BASE_MODEL_OUTPUT_DIR = "trained_bkg_models_timestamped_v2"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_DATALOADER_WORKERS = 0
SEED = 42

RAW_IMAGE_BRANCH_NAMES = ["raw_image_u", "raw_image_v", "raw_image_w"]
RECO_IMAGE_BRANCH_NAMES = ["reco_image_u", "reco_image_v", "reco_image_w"]
EVENT_CATEGORY_BRANCH_NAME = "event_category"

IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512
RESNET_INPUT_CHANNELS = 2 

MUON_NEUTRINO_CC_BACKGROUND_CATEGORIES = [20, 21, 100, 101, 102, 103, 104, 105, 106]
CONTRASTIVE_PRETRAINING_CATEGORIES = MUON_NEUTRINO_CC_BACKGROUND_CATEGORIES
SIGNAL_CATEGORIES_TO_EXCLUDE_ENTIRELY = [0, 1, 2, 10, 11]


CONTRASTIVE_RESNET_VARIANT = 'resnet50'
CONTRASTIVE_FINAL_EMBEDDING_DIM = 128
CONTRASTIVE_PROJECTION_HEAD_HIDDEN_DIM = 512
CONTRASTIVE_PROJECTION_DIM = 128
CONTRASTIVE_LEARNING_RATE = 3e-4
CONTRASTIVE_BATCH_SIZE = 16
EFFECTIVE_BATCH_SIZE = 256
GRADIENT_ACCUMULATION_STEPS = EFFECTIVE_BATCH_SIZE // CONTRASTIVE_BATCH_SIZE if CONTRASTIVE_BATCH_SIZE > 0 else 1
CONTRASTIVE_NUM_EPOCHS = 0
CONTRASTIVE_WARMUP_EPOCHS = 5
CONTRASTIVE_TEMPERATURE = 0.07
CONTRASTIVE_EARLY_STOPPING_PATIENCE = 10
CONTRASTIVE_MIN_DELTA = 0.001
PRETRAINED_ENCODER_PATH_FOR_STAGE2 = None 

BKG_ISOLATION_CLASSIFIER_LEARNING_RATE = 1e-3
BKG_ISOLATION_CLASSIFIER_BATCH_SIZE = 18
BKG_ISOLATION_CLASSIFIER_NUM_EPOCHS = 30
CLASSIFIER_EARLY_STOPPING_PATIENCE = 5
CLASSIFIER_MIN_DELTA = 0.005
CLASSIFIER_LR_SCHEDULER_PATIENCE = 3
CLASSIFIER_LR_SCHEDULER_FACTOR = 0.2
PRINT_VALIDATION_BATCH_METRICS = True

# --- Setup ---
os.makedirs(BASE_MODEL_OUTPUT_DIR, exist_ok=True)
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED); torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = True

def get_timestamp():
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_UTC")

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

class RootContrastiveTripletDataset(Dataset): 
    def __init__(self, root_file_path, tree_name, raw_branch_names, reco_branch_names,
                 event_category_branch, image_height, image_width,
                 categories_to_use_for_training, signal_categories_to_avoid):
        super().__init__()
        self.root_file_path = root_file_path; self.tree_name = tree_name
        self.raw_branch_names = raw_branch_names
        self.reco_branch_names = reco_branch_names
        self.num_planes = len(raw_branch_names)
        if self.num_planes != 3:
            raise ValueError(f"RootContrastiveTripletDataset expects 3 planes, got {self.num_planes}")
        self.event_category_branch = event_category_branch
        self.image_height = image_height; self.image_width = image_width
        self.categories_to_use_for_training = set(categories_to_use_for_training)
        self.signal_categories_to_avoid = set(signal_categories_to_avoid)
        self.event_indices = []; self.root_file_handle = None
        self.all_branches_to_fetch_once = []
        for plane_idx in range(self.num_planes):
            self.all_branches_to_fetch_once.append(self.raw_branch_names[plane_idx])
            self.all_branches_to_fetch_once.append(self.reco_branch_names[plane_idx])

        try: self._open_root_file(); self._prepare_indices()
        except Exception as e: print(f"CRITICAL ERROR: ContrastiveTripletDataset Init Error: {e}", flush=True); self._close_root_file(); raise

    def _open_root_file(self):
        if not self.root_file_handle: self.root_file_handle = uproot.open(self.root_file_path); self.tree = self.root_file_handle[self.tree_name]

    def _close_root_file(self):
        if self.root_file_handle: self.root_file_handle.close(); self.root_file_handle = None

    def close(self): self._close_root_file()

    def _prepare_indices(self):
        print(f"INFO: ContrastiveTripletDataset: Preparing event indices for triplets. Using: {self.categories_to_use_for_training}. Avoiding: {self.signal_categories_to_avoid}", flush=True)
        event_categories_data = self.tree[self.event_category_branch].array(library="ak"); num_events = len(event_categories_data)
        for event_idx in range(num_events):
            event_cat = event_categories_data[event_idx]
            if event_cat in self.signal_categories_to_avoid or event_cat not in self.categories_to_use_for_training:
                continue
            self.event_indices.append(event_idx)
        print(f"INFO: ContrastiveTripletDataset (Full): Prepared {len(self.event_indices)} events for triplet generation.", flush=True)

    def __len__(self): return len(self.event_indices)

    def __getitem__(self, idx):
        if not self.root_file_handle: self._open_root_file()
        event_idx = self.event_indices[idx]
        empty_tensor_set = (torch.empty(0), torch.empty(0), torch.empty(0))

        try:
            event_data = self.tree.arrays(self.all_branches_to_fetch_once, entry_start=event_idx, entry_stop=event_idx + 1, library="ak")
            views = []
            for plane_idx in range(self.num_planes):
                raw_branch = self.raw_branch_names[plane_idx]
                reco_branch = self.reco_branch_names[plane_idx]

                raw_img_data = ak.to_numpy(event_data[raw_branch][0])
                reco_img_data = ak.to_numpy(event_data[reco_branch][0])

                raw_img = process_single_plane_image_data(raw_img_data, self.image_height, self.image_width)
                reco_img = process_single_plane_image_data(reco_img_data, self.image_height, self.image_width)
                view_tensor = torch.cat((raw_img, reco_img), dim=0) # Shape: (2, H, W)
                views.append(view_tensor)

            if len(views) != 3: 
                print(f"WARNING: ContrastiveTripletDataset: Expected 3 views, got {len(views)} for event_idx {event_idx}", flush=True)
                return empty_tensor_set 
            return tuple(views) 

        except Exception as e:
            print(f"ERROR: ContrastiveTripletDataset: GetItem Error for entry {idx} (event {event_idx}): {e}\n{traceback.format_exc()}", flush=True)
            return empty_tensor_set

class RootBinaryClassifierDataset(Dataset): # MODIFIED
    def __init__(self, root_file_path, tree_name, raw_image_branch_names_ordered, reco_image_branch_names_ordered, event_category_branch,
                 positive_event_categories, negative_event_categories, image_height, image_width,
                 dataset_name="Classifier", signal_categories_to_avoid_in_dataset=None):
        super().__init__()
        self.root_file_path, self.tree_name = root_file_path, tree_name
        self.raw_image_branch_names = raw_image_branch_names_ordered
        self.reco_image_branch_names = reco_image_branch_names_ordered
        self.num_plane_types = len(raw_image_branch_names_ordered)
        self.event_category_branch = event_category_branch
        self.image_height = image_height
        self.image_width = image_width
        self.positive_categories = set(positive_event_categories)
        self.negative_categories = set(negative_event_categories)
        self.signal_categories_to_avoid_in_dataset = set(signal_categories_to_avoid_in_dataset) if signal_categories_to_avoid_in_dataset else set()
        self.dataset_name = dataset_name
        self.event_plane_info_list = [] # Stores (event_idx, plane_type_idx, label)
        self.root_file_handle = None
        self.all_branches_to_fetch_map = {}

        try:
            self._open_root_file()
            self._prepare_indices()
        except Exception as e:
            print(f"CRITICAL ERROR: {self.dataset_name} Dataset: Init Error: {e}", flush=True)
            if self.root_file_handle: self._close_root_file()
            raise

    def _open_root_file(self):
        if not self.root_file_handle: self.root_file_handle = uproot.open(self.root_file_path)
        self.tree = self.root_file_handle[self.tree_name]

    def _close_root_file(self):
        if self.root_file_handle:
            self.root_file_handle.close()
            self.root_file_handle = None
    def close(self): self._close_root_file()

    def _prepare_indices(self): # Same as before
        print(f"INFO: {self.dataset_name} Dataset (Full): Preparing. Positive: {self.positive_categories}, Negative: {self.negative_categories}, Avoiding: {self.signal_categories_to_avoid_in_dataset}", flush=True)
        event_categories_data = self.tree[self.event_category_branch].array(library="ak"); num_pos_events, num_neg_events = 0,0; unique_events_processed = set()
        for event_idx, event_cat_val in enumerate(event_categories_data):
            if event_cat_val in self.signal_categories_to_avoid_in_dataset: continue
            label = -1; is_pos = event_cat_val in self.positive_categories; is_neg = event_cat_val in self.negative_categories
            if is_pos: label = 1
            elif is_neg: label = 0
            else: continue # Skip if not in positive or negative categories for this classifier
            if event_idx not in unique_events_processed:
                if is_pos: num_pos_events +=1
                if is_neg: num_neg_events +=1
                unique_events_processed.add(event_idx)
            # Create an item for each plane type (U, V, W) for this event
            for plane_type_idx in range(self.num_plane_types):
                self.event_plane_info_list.append((event_idx, plane_type_idx, label))
        print(f"INFO: {self.dataset_name} Dataset (Full): Prepared {len(self.event_plane_info_list)} items (plane-wise). Unique Positive Events: {num_pos_events}, Unique Negative Events: {num_neg_events}.", flush=True)


    def __len__(self): return len(self.event_plane_info_list)

    def __getitem__(self, idx): # MODIFIED
        if not self.root_file_handle: self._open_root_file()
        actual_event_idx, plane_type_idx, label = self.event_plane_info_list[idx]
        raw_branch_name = self.raw_image_branch_names[plane_type_idx]
        reco_branch_name = self.reco_image_branch_names[plane_type_idx]
        # Returns ( (2,H,W) CPU tensor, scalar CPU tensor )
        empty_item = (torch.empty(0, self.image_height, self.image_width), torch.empty(0))


        if plane_type_idx not in self.all_branches_to_fetch_map:
            self.all_branches_to_fetch_map[plane_type_idx] = [raw_branch_name, reco_branch_name]
        branches_to_fetch = self.all_branches_to_fetch_map[plane_type_idx]

        try:
            event_data = self.tree.arrays(branches_to_fetch, entry_start=actual_event_idx, entry_stop=actual_event_idx + 1, library="ak")
            raw_img_data_np = ak.to_numpy(event_data[raw_branch_name][0])
            reco_img_data_np = ak.to_numpy(event_data[reco_branch_name][0])

            # process_single_plane_image_data returns a (1, H, W) CPU tensor
            raw_img_tensor = process_single_plane_image_data(raw_img_data_np, self.image_height, self.image_width)
            reco_img_tensor = process_single_plane_image_data(reco_img_data_np, self.image_height, self.image_width)

            # Concatenate along the channel dimension (dim=0 for CHW)
            combined_image_tensor = torch.cat((raw_img_tensor, reco_img_tensor), dim=0) # Shape: (2, H, W) on CPU

            label_tensor = torch.tensor(label, dtype=torch.float32) # CPU tensor
            return combined_image_tensor, label_tensor

        except Exception as e:
            print(f"ERROR: {self.dataset_name} Dataset: GetItem Error idx {idx} (event {actual_event_idx}, plane {plane_type_idx}): {e}\n{traceback.format_exc()}", flush=True)
            return empty_item

# ResNetEncoder, ProjectionHead, ContrastiveTripletNetwork, NTXentLoss, TripletNTXentLoss are unchanged
class ResNetEncoder(nn.Module):
    def __init__(self, final_embedding_dim, resnet_variant='resnet18', input_channels=2):
        super().__init__(); current_weights = None
        if resnet_variant == 'resnet18': resnet = models.resnet18(weights=current_weights)
        elif resnet_variant == 'resnet34': resnet = models.resnet34(weights=current_weights)
        elif resnet_variant == 'resnet50': resnet = models.resnet50(weights=current_weights)
        else: raise ValueError(f"Unsupported ResNet variant: {resnet_variant}")
        if input_channels != 3:
            original_conv1 = resnet.conv1
            resnet.conv1 = nn.Conv2d(input_channels, original_conv1.out_channels, kernel_size=original_conv1.kernel_size, stride=original_conv1.stride, padding=original_conv1.padding, bias=False)
        self.features_dim = resnet.fc.in_features; self.base_encoder = nn.Sequential(*list(resnet.children())[:-1])
        self.final_fc = nn.Linear(self.features_dim, final_embedding_dim)
    def forward(self, x): x = self.base_encoder(x); x = torch.flatten(x, 1); x = self.final_fc(x); return x

class ProjectionHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__(); self.network = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim))
    def forward(self, x): return F.normalize(self.network(x), p=2, dim=1)

class ContrastiveTripletNetwork(nn.Module):
    def __init__(self, encoder, projection_head):
        super().__init__(); self.encoder = encoder; self.projection_head = projection_head
    def forward(self, view_u, view_v, view_w):
        emb_u = self.encoder(view_u); emb_v = self.encoder(view_v); emb_w = self.encoder(view_w)
        proj_u = self.projection_head(emb_u); proj_v = self.projection_head(emb_v); proj_w = self.projection_head(emb_w)
        return proj_u, proj_v, proj_w

class NTXentLoss(nn.Module):
    def __init__(self, temperature, device):
        super().__init__(); self.temperature = temperature; self.device = device
        self.criterion = nn.CrossEntropyLoss(reduction="sum"); self.similarity_function = nn.CosineSimilarity(dim=2)
    def forward(self, proj_i, proj_j):
        batch_size = proj_i.shape[0]
        if batch_size == 0: return torch.tensor(0.0, device=self.device, requires_grad=True)
        representations = torch.cat([proj_i, proj_j], dim=0)
        sim_matrix = self.similarity_function(representations.unsqueeze(1), representations.unsqueeze(0))
        mask_self_similarity = torch.eye(2 * batch_size, dtype=torch.bool, device=self.device)
        sim_matrix = sim_matrix.masked_fill(mask_self_similarity, -float('inf')); sim_matrix = sim_matrix / self.temperature
        labels_row1 = torch.arange(batch_size, 2 * batch_size, device=self.device); labels_row2 = torch.arange(batch_size, device=self.device)
        labels = torch.cat([labels_row1, labels_row2]).long(); loss = self.criterion(sim_matrix, labels); return loss / (2 * batch_size)

class TripletNTXentLoss(nn.Module):
    def __init__(self, temperature, device):
        super().__init__(); self.pairwise_ntxent_loss = NTXentLoss(temperature, device)
    def forward(self, proj_u, proj_v, proj_w):
        loss_uv = self.pairwise_ntxent_loss(proj_u, proj_v)
        loss_uw = self.pairwise_ntxent_loss(proj_u, proj_w)
        loss_vw = self.pairwise_ntxent_loss(proj_v, proj_w)
        return (loss_uv + loss_uw + loss_vw) / 3.0
# SimpleCNN is unchanged, but its instantiation will use different parameters
class SimpleCNN(nn.Module):
    def __init__(self, input_channels, map_h, map_w, num_classes=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Calculate flat size after one conv and one pool
        # Assuming kernel_size=2, stride=2 for pool1
        pooled_height = map_h // 2
        pooled_width = map_w // 2
        # Ensure pooled dimensions are at least 1
        pooled_height = max(1, pooled_height)
        pooled_width = max(1, pooled_width)

        self.flat_size = 16 * pooled_height * pooled_width
        self.fc1 = nn.Linear(self.flat_size, num_classes)

    def forward(self, x):
        # print(f"SimpleCNN input shape: {x.shape}") # Debug
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        # print(f"SimpleCNN after pool shape: {x.shape}") # Debug
        x = x.view(-1, self.flat_size)
        # print(f"SimpleCNN after flatten shape: {x.shape}") # Debug
        x = self.fc1(x)
        # print(f"SimpleCNN output shape: {x.shape}") # Debug
        return x

# get_simclr_scheduler, train_contrastive_network, train_binary_classifier are unchanged
# (Make sure train_binary_classifier correctly handles the input shape for tracing)
def get_simclr_scheduler(optimizer, warmup_epochs, total_epochs, initial_lr_base):
    warmup_epochs = max(1, warmup_epochs); total_epochs = max(total_epochs, warmup_epochs)
    def lr_lambda(current_epoch):
        if current_epoch < warmup_epochs: return float(current_epoch + 1) / float(warmup_epochs)
        else:
            progress = float(current_epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
            return 0.5 * (1.0 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)

def train_contrastive_network(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs, device,
                              model_output_dir_ts,
                              base_filename_prefix,
                              losses_basename,
                              example_input_shape_for_trace, # e.g. (RESNET_INPUT_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH)
                              early_stopping_patience, min_delta,
                              gradient_accumulation_steps=1):
    print(f"INFO: Starting Triplet SimCLR Pre-training. Output Dir: {model_output_dir_ts}", flush=True)
    actual_gradient_accumulation_steps = max(1, gradient_accumulation_steps)
    print(f"Effective Batch Size: {CONTRASTIVE_BATCH_SIZE * actual_gradient_accumulation_steps} (Mini-Batch: {CONTRASTIVE_BATCH_SIZE}, Accumulation Steps: {actual_gradient_accumulation_steps})", flush=True)
    print(f"Early Stopping: Patience={early_stopping_patience}, Min Delta={min_delta}", flush=True)

    best_model_save_path = os.path.join(model_output_dir_ts, f"{base_filename_prefix}_best.pth")
    final_state_dict_path = os.path.join(model_output_dir_ts, f"{base_filename_prefix}_final.pth")
    final_torchscript_path = os.path.join(model_output_dir_ts, f"{base_filename_prefix}_final.pt")
    losses_npz_path = os.path.join(model_output_dir_ts, f"{losses_basename}.npz")

    all_train_batch_losses = []
    epoch_metrics = {'avg_train_loss': [], 'avg_val_loss': [], 'lr': []}

    amp_enabled = (device.type == 'cuda')
    scaler = GradScaler(enabled=amp_enabled)
    best_val_loss, epochs_no_improve = float('inf'), 0

    for epoch in range(epochs):
        model.train()
        current_epoch_train_loss_sum, num_optimizer_steps_this_epoch = 0.0, 0
        # Calculate print_freq_train safely
        num_total_optimizer_steps_per_epoch = (len(train_loader) + actual_gradient_accumulation_steps - 1) // actual_gradient_accumulation_steps if actual_gradient_accumulation_steps > 0 else len(train_loader)
        print_freq_train = max(1, num_total_optimizer_steps_per_epoch // 10) if num_total_optimizer_steps_per_epoch > 0 else 1


        optimizer.zero_grad(set_to_none=True)

        for batch_idx, views_train_tuple in enumerate(train_loader):
            view_u_train, view_v_train, view_w_train = views_train_tuple
            if view_u_train.nelement() == 0 or view_v_train.nelement() == 0 or view_w_train.nelement() == 0 : continue # check all views
            view_u_train, view_v_train, view_w_train = view_u_train.to(device), view_v_train.to(device), view_w_train.to(device)

            with torch.cuda.amp.autocast(enabled=amp_enabled):
                proj_u_train, proj_v_train, proj_w_train = model(view_u_train, view_v_train, view_w_train)
                train_loss_batch = criterion(proj_u_train, proj_v_train, proj_w_train)
                if actual_gradient_accumulation_steps > 1:
                    train_loss_batch = train_loss_batch / actual_gradient_accumulation_steps

            if amp_enabled:
                scaler.scale(train_loss_batch).backward()
            else:
                train_loss_batch.backward()

            # Log the original scale of the loss (sum of losses in accumulation window before averaging)
            all_train_batch_losses.append(train_loss_batch.item() * (actual_gradient_accumulation_steps if actual_gradient_accumulation_steps > 1 else 1.0))
            # Accumulate sum of original scale losses
            current_epoch_train_loss_sum += train_loss_batch.item() * (actual_gradient_accumulation_steps if actual_gradient_accumulation_steps > 1 else 1.0)


            if (batch_idx + 1) % actual_gradient_accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                if amp_enabled:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                num_optimizer_steps_this_epoch += 1
                if num_optimizer_steps_this_epoch > 0 and print_freq_train > 0 and num_optimizer_steps_this_epoch % print_freq_train == 0:
                    # Calculate average loss for the completed accumulation cycle for printing
                    # This logic needs to be careful: current_epoch_train_loss_sum is cumulative for the epoch.
                    # We want average loss over the last 'print_freq_train' optimizer steps.
                    # Simpler: print the running average for the epoch so far.
                    avg_loss_so_far_this_epoch = current_epoch_train_loss_sum / ( (batch_idx + 1) * CONTRASTIVE_BATCH_SIZE / (CONTRASTIVE_BATCH_SIZE * actual_gradient_accumulation_steps) ) if (batch_idx +1) > 0 else 0
                    avg_loss_for_print_cycle = current_epoch_train_loss_sum / num_optimizer_steps_this_epoch if num_optimizer_steps_this_epoch > 0 else 0
                    print(f"  Epoch [{epoch+1}/{epochs}] Opt Step [{num_optimizer_steps_this_epoch}/{num_total_optimizer_steps_per_epoch}], Avg Loss in Cycle: {avg_loss_for_print_cycle:.4f}", flush=True)


        # Avg train loss per optimizer step for the epoch
        avg_epoch_train_loss = current_epoch_train_loss_sum / num_optimizer_steps_this_epoch if num_optimizer_steps_this_epoch > 0 else float('nan')
        epoch_metrics['avg_train_loss'].append(avg_epoch_train_loss)


        current_epoch_avg_val_loss = float('inf')
        if val_loader and len(val_loader) > 0:
            model.eval()
            current_epoch_val_loss_sum, num_val_batches = 0.0, 0
            with torch.no_grad():
                for views_val_tuple in val_loader:
                    view_u_val, view_v_val, view_w_val = views_val_tuple
                    if view_u_val.nelement() == 0 or view_v_val.nelement() == 0 or view_w_val.nelement() == 0: continue
                    view_u_val, view_v_val, view_w_val = view_u_val.to(device), view_v_val.to(device), view_w_val.to(device)
                    with torch.cuda.amp.autocast(enabled=amp_enabled):
                        proj_u_val, proj_v_val, proj_w_val = model(view_u_val, view_v_val, view_w_val)
                        val_loss_batch = criterion(proj_u_val, proj_v_val, proj_w_val)
                    current_epoch_val_loss_sum += val_loss_batch.item()
                    num_val_batches += 1
            current_epoch_avg_val_loss = current_epoch_val_loss_sum / num_val_batches if num_val_batches > 0 else float('inf')
        epoch_metrics['avg_val_loss'].append(current_epoch_avg_val_loss)
        epoch_metrics['lr'].append(optimizer.param_groups[0]['lr'])

        print(f"Contrastive Epoch [{epoch+1}/{epochs}] Completed. Avg Train Loss (per opt step): {avg_epoch_train_loss:.4f}, Avg Val Loss: {current_epoch_avg_val_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.3e}", flush=True)

        if scheduler:
            if isinstance(scheduler, ReduceLROnPlateau): scheduler.step(current_epoch_avg_val_loss)
            else: scheduler.step()

        np.savez(losses_npz_path,
                 all_train_batch_losses=np.array(all_train_batch_losses),
                 epoch_avg_train_loss=np.array(epoch_metrics['avg_train_loss']),
                 epoch_avg_val_loss=np.array(epoch_metrics['avg_val_loss']),
                 epoch_lr=np.array(epoch_metrics['lr']),
                 last_saved_timestamp_utc=datetime.now(timezone.utc).isoformat())

        if current_epoch_avg_val_loss < best_val_loss - min_delta:
            best_val_loss = current_epoch_avg_val_loss; epochs_no_improve = 0
            if hasattr(model, 'encoder'): # Save encoder if it's part of the model (e.g. ContrastiveTripletNetwork)
                 torch.save(model.encoder.state_dict(), best_model_save_path)
            else: # Save the whole model if no 'encoder' attribute
                 torch.save(model.state_dict(), best_model_save_path)
            print(f"  Val loss improved to {best_val_loss:.4f}. Saved best model to {best_model_save_path}", flush=True)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stopping_patience:
                print(f"  Early stopping triggered after {epochs_no_improve} epochs of no val_loss improvement.", flush=True)
                break

    print("Contrastive pre-training finished.", flush=True)
    final_encoder_state_dict_to_save = None
    if hasattr(model, 'encoder'):
        if os.path.exists(best_model_save_path):
            print(f"Loading best encoder model from {best_model_save_path} for final save.", flush=True)
            # Create a new encoder instance to load the state dict into, to avoid issues with model wrapper
            temp_encoder = ResNetEncoder(CONTRASTIVE_FINAL_EMBEDDING_DIM, CONTRASTIVE_RESNET_VARIANT, RESNET_INPUT_CHANNELS)
            temp_encoder.load_state_dict(torch.load(best_model_save_path, map_location=DEVICE))
            final_encoder_state_dict_to_save = temp_encoder.state_dict()
            model.encoder.load_state_dict(final_encoder_state_dict_to_save) # also update the main model's encoder
        else:
            print(f"Warning: Best encoder model {best_model_save_path} not found. Using last encoder state for final save.", flush=True)
            final_encoder_state_dict_to_save = model.encoder.state_dict()
        torch.save(final_encoder_state_dict_to_save, final_state_dict_path)
        print(f"Final contrastive encoder state_dict saved to {final_state_dict_path}", flush=True)
        try:
            encoder_to_trace = ResNetEncoder(CONTRASTIVE_FINAL_EMBEDDING_DIM, CONTRASTIVE_RESNET_VARIANT, RESNET_INPUT_CHANNELS).to(DEVICE)
            encoder_to_trace.load_state_dict(final_encoder_state_dict_to_save)
            encoder_to_trace.eval()
            example_input = torch.randn(1, *example_input_shape_for_trace, device=device)
            traced_encoder = torch.jit.trace(encoder_to_trace, example_input)
            traced_encoder.save(final_torchscript_path)
            print(f"Final contrastive encoder TorchScript saved to {final_torchscript_path}", flush=True)
        except Exception as e: print(f"ERROR saving final TorchScript encoder: {e}\n{traceback.format_exc()}", flush=True)

    else: # if model has no .encoder attribute, save the whole model
        if os.path.exists(best_model_save_path): # This path would save the whole model's state_dict
            model.load_state_dict(torch.load(best_model_save_path, map_location=DEVICE))
        torch.save(model.state_dict(), final_state_dict_path)
        print(f"Final model state_dict saved to {final_state_dict_path}", flush=True)
        # Tracing for a generic model might need more careful example input
        # try:
        #     model.eval()
        #     example_input = torch.randn(1, *example_input_shape_for_trace, device=device) # This shape might be wrong for a generic model
        #     traced_model = torch.jit.trace(model, example_input)
        #     traced_model.save(final_torchscript_path)
        #     print(f"Final model TorchScript saved to {final_torchscript_path}", flush=True)
        # except Exception as e: print(f"ERROR saving final TorchScript model: {e}", flush=True)


    print(f"Final loss data for contrastive stage saved to {losses_npz_path}", flush=True)


def train_binary_classifier(classifier_name, cnn_model, train_loader, val_loader, criterion, optimizer, scheduler, epochs, device,
                            model_output_dir_ts,
                            base_filename_prefix,
                            losses_basename,
                            cnn_input_shape_for_trace, # e.g. (RESNET_INPUT_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH)
                            early_stopping_patience, min_delta,
                            print_val_batch_metrics=False):
    print(f"INFO: Training {classifier_name}. Output Dir: {model_output_dir_ts}", flush=True)
    print(f"Early Stopping: Patience={early_stopping_patience}, Min Delta={min_delta}", flush=True)
    print(f"Input shape for JIT trace: {cnn_input_shape_for_trace}", flush=True)


    best_model_save_path = os.path.join(model_output_dir_ts, f"{base_filename_prefix}_best.pth")
    final_state_dict_path = os.path.join(model_output_dir_ts, f"{base_filename_prefix}_final.pth")
    final_torchscript_path = os.path.join(model_output_dir_ts, f"{base_filename_prefix}_final.pt")
    losses_npz_path = os.path.join(model_output_dir_ts, f"{losses_basename}.npz")

    all_train_batch_losses, all_train_batch_accuracies = [], []
    all_val_batch_losses, all_val_batch_accuracies = [], []
    epoch_summary_metrics = {'avg_train_loss': [], 'train_acc': [], 'avg_val_loss': [], 'val_acc': [], 'lr': []}

    amp_enabled = (device.type == 'cuda')
    scaler = GradScaler(enabled=amp_enabled)
    best_val_loss, epochs_no_improve = float('inf'), 0

    for epoch in range(epochs):
        cnn_model.train()
        current_epoch_train_loss_sum, num_train_batches_processed = 0.0, 0
        current_epoch_train_correct_preds, current_epoch_train_total_samples = 0, 0
        # Calculate print_freq_train_cls safely
        print_freq_train_cls = max(1, len(train_loader) // 10) if len(train_loader) > 0 else 1


        for batch_idx, (features_train, labels_train) in enumerate(train_loader):
            if features_train.nelement() == 0: continue
            features_train, labels_train = features_train.to(device), labels_train.to(device).unsqueeze(1)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                outputs_train = cnn_model(features_train)
                train_loss_batch = criterion(outputs_train, labels_train)

            if amp_enabled: scaler.scale(train_loss_batch).backward(); scaler.step(optimizer); scaler.update()
            else: train_loss_batch.backward(); optimizer.step()

            train_loss_batch_val = train_loss_batch.item()
            preds_train_batch = torch.sigmoid(outputs_train.detach().float()) > 0.5 # Use .detach() for preds if outputs_train requires grad
            train_acc_batch_val = (preds_train_batch == labels_train.bool()).sum().item() / labels_train.size(0) if labels_train.size(0) > 0 else 0.0


            all_train_batch_losses.append(train_loss_batch_val)
            all_train_batch_accuracies.append(train_acc_batch_val)
            current_epoch_train_loss_sum += train_loss_batch_val
            current_epoch_train_correct_preds += (preds_train_batch == labels_train.bool()).sum().item()
            current_epoch_train_total_samples += labels_train.size(0)
            num_train_batches_processed +=1

            if print_freq_train_cls > 0 and batch_idx > 0 and batch_idx % print_freq_train_cls == 0 :
                print(f"  {classifier_name} Epoch [{epoch+1}/{epochs}] Train Batch [{batch_idx}/{len(train_loader)}], Loss: {train_loss_batch_val:.4f}, Acc: {train_acc_batch_val:.4f}", flush=True)

        avg_epoch_train_loss = current_epoch_train_loss_sum / num_train_batches_processed if num_train_batches_processed > 0 else float('nan')
        avg_epoch_train_acc = current_epoch_train_correct_preds / current_epoch_train_total_samples if current_epoch_train_total_samples > 0 else float('nan')
        epoch_summary_metrics['avg_train_loss'].append(avg_epoch_train_loss)
        epoch_summary_metrics['train_acc'].append(avg_epoch_train_acc)

        current_epoch_avg_val_loss, current_epoch_avg_val_acc = float('inf'), float('nan')
        if val_loader and len(val_loader) > 0:
            cnn_model.eval()
            current_epoch_val_loss_sum, num_val_batches_processed = 0.0, 0
            current_epoch_val_correct_preds, current_epoch_val_total_samples = 0, 0
            # Calculate print_freq_val_cls safely
            print_freq_val_cls = max(1, len(val_loader) // 5) if len(val_loader) > 0 else 1


            with torch.no_grad():
                for val_batch_idx, (features_val, labels_val) in enumerate(val_loader):
                    if features_val.nelement() == 0: continue
                    features_val, labels_val = features_val.to(device), labels_val.to(device).unsqueeze(1)
                    with torch.cuda.amp.autocast(enabled=amp_enabled):
                        outputs_val = cnn_model(features_val)
                        val_loss_batch = criterion(outputs_val, labels_val)

                    val_loss_batch_val = val_loss_batch.item()
                    preds_val_batch = torch.sigmoid(outputs_val.float()) > 0.5
                    val_acc_batch_val = (preds_val_batch == labels_val.bool()).sum().item() / labels_val.size(0) if labels_val.size(0) > 0 else 0.0

                    all_val_batch_losses.append(val_loss_batch_val)
                    all_val_batch_accuracies.append(val_acc_batch_val)
                    current_epoch_val_loss_sum += val_loss_batch_val
                    current_epoch_val_correct_preds += (preds_val_batch == labels_val.bool()).sum().item()
                    current_epoch_val_total_samples += labels_val.size(0)
                    num_val_batches_processed +=1

                    if print_val_batch_metrics and print_freq_val_cls > 0 and val_batch_idx > 0 and val_batch_idx % print_freq_val_cls == 0 :
                        print(f"  {classifier_name} Epoch [{epoch+1}/{epochs}] Val Batch [{val_batch_idx}/{len(val_loader)}], Loss: {val_loss_batch_val:.4f}, Acc: {val_acc_batch_val:.4f}", flush=True)

            current_epoch_avg_val_loss = current_epoch_val_loss_sum / num_val_batches_processed if num_val_batches_processed > 0 else float('inf')
            current_epoch_avg_val_acc = current_epoch_val_correct_preds / current_epoch_val_total_samples if current_epoch_val_total_samples > 0 else float('nan')
        epoch_summary_metrics['avg_val_loss'].append(current_epoch_avg_val_loss)
        epoch_summary_metrics['val_acc'].append(current_epoch_avg_val_acc)
        epoch_summary_metrics['lr'].append(optimizer.param_groups[0]['lr'])

        print(f"{classifier_name} Epoch [{epoch+1}/{epochs}] COMPLETED. Train L/Acc: {avg_epoch_train_loss:.4f}/{avg_epoch_train_acc:.4f} || Val L/Acc: {current_epoch_avg_val_loss:.4f}/{current_epoch_avg_val_acc:.4f} || LR: {optimizer.param_groups[0]['lr']:.3e}", flush=True)

        if scheduler:
            if isinstance(scheduler, ReduceLROnPlateau): scheduler.step(current_epoch_avg_val_loss)
            else: scheduler.step()

        np.savez(losses_npz_path,
                 all_train_batch_losses=np.array(all_train_batch_losses), all_train_batch_accuracies=np.array(all_train_batch_accuracies),
                 all_val_batch_losses=np.array(all_val_batch_losses), all_val_batch_accuracies=np.array(all_val_batch_accuracies),
                 epoch_avg_train_loss=np.array(epoch_summary_metrics['avg_train_loss']), epoch_train_acc=np.array(epoch_summary_metrics['train_acc']),
                 epoch_avg_val_loss=np.array(epoch_summary_metrics['avg_val_loss']), epoch_val_acc=np.array(epoch_summary_metrics['val_acc']),
                 epoch_lr=np.array(epoch_summary_metrics['lr']), last_saved_timestamp_utc=datetime.now(timezone.utc).isoformat())

        if current_epoch_avg_val_loss < best_val_loss - min_delta:
            best_val_loss = current_epoch_avg_val_loss; epochs_no_improve = 0
            torch.save(cnn_model.state_dict(), best_model_save_path)
            print(f"  Val loss improved to {best_val_loss:.4f}. Saved best {classifier_name} model to {best_model_save_path}", flush=True)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stopping_patience:
                print(f"  Early stopping for {classifier_name} triggered after {epochs_no_improve} epochs.", flush=True); break

    print(f"{classifier_name} training finished.", flush=True)
    if os.path.exists(best_model_save_path):
        print(f"Loading best {classifier_name} model from {best_model_save_path} for final save.", flush=True)
        cnn_model.load_state_dict(torch.load(best_model_save_path, map_location=DEVICE))
    else: print(f"Warning: Best model for {classifier_name} ({best_model_save_path}) not found. Using last model state for final save.", flush=True)

    torch.save(cnn_model.state_dict(), final_state_dict_path)
    print(f"Final {classifier_name} state_dict saved to {final_state_dict_path}", flush=True)
    try:
        cnn_model.eval()
        example_input_cnn = torch.randn(1, *cnn_input_shape_for_trace, device=device) # Use the passed shape
        traced_cnn = torch.jit.trace(cnn_model, example_input_cnn); traced_cnn.save(final_torchscript_path)
        print(f"Final {classifier_name} TorchScript saved to {final_torchscript_path}", flush=True)
    except Exception as e: print(f"ERROR saving final TorchScript for {classifier_name}: {e}\n{traceback.format_exc()}", flush=True)
    print(f"Final loss data for {classifier_name} saved to {losses_npz_path}", flush=True)

# --- Main Execution ---
if __name__ == "__main__":
    main_run_timestamp = get_timestamp()
    print(f"INFO: Main run timestamp: {main_run_timestamp}", flush=True)
    print(f"INFO: Device: {DEVICE}. PyTorch Version: {torch.__version__}", flush=True)
    print(f"INFO: ResNet Encoder Input C,H,W: {RESNET_INPUT_CHANNELS},{IMAGE_HEIGHT},{IMAGE_WIDTH}. Encoder Emb Dim: {CONTRASTIVE_FINAL_EMBEDDING_DIM}", flush=True)
    print(f"INFO: SimpleCNN (Binary Classifier) Input C,H,W: {RESNET_INPUT_CHANNELS},{IMAGE_HEIGHT},{IMAGE_WIDTH}", flush=True)
    print(f"INFO: Contrastive training epochs (Stage 1): {CONTRASTIVE_NUM_EPOCHS}", flush=True)
    print(f"INFO: Classifier training epochs (Stage 2): {BKG_ISOLATION_CLASSIFIER_NUM_EPOCHS}", flush=True)


    # Stage 1: Contrastive Pre-training (for ResNet Encoder)
    encoder_path_from_stage1_run = None # Path to the encoder .pth file if Stage 1 runs
    if CONTRASTIVE_NUM_EPOCHS > 0:
        stg1_run_ts = get_timestamp()
        stg1_output_dir = os.path.join(BASE_MODEL_OUTPUT_DIR, f"S1_ContrastivePretrain_{stg1_run_ts}")
        os.makedirs(stg1_output_dir, exist_ok=True)
        print(f"\n--- Stage 1: Contrastive Triplet ResNet Encoder Pre-training (Run ID: {stg1_run_ts}) ---", flush=True)
        stg1_model_base_filename = "bkg_resnet_encoder_triplet"
        stg1_losses_basename = "contrastive_run_metrics"
        full_stg1_dataset = None
        try:
            full_stg1_dataset = RootContrastiveTripletDataset( ROOT_FILE_PATH, TREE_NAME, RAW_IMAGE_BRANCH_NAMES, RECO_IMAGE_BRANCH_NAMES, EVENT_CATEGORY_BRANCH_NAME, IMAGE_HEIGHT, IMAGE_WIDTH, CONTRASTIVE_PRETRAINING_CATEGORIES, SIGNAL_CATEGORIES_TO_EXCLUDE_ENTIRELY )
            if len(full_stg1_dataset) > 0:
                val_ratio = 0.2
                actual_batch_size_stg1 = CONTRASTIVE_BATCH_SIZE if CONTRASTIVE_BATCH_SIZE > 0 else 1
                min_val_samples = actual_batch_size_stg1 * 2
                current_total_samples = len(full_stg1_dataset)
                val_size = 0
                if current_total_samples * val_ratio < min_val_samples and current_total_samples > actual_batch_size_stg1 * 4 :
                     val_size = min_val_samples
                elif current_total_samples * val_ratio >= min_val_samples:
                     val_size = int(current_total_samples * val_ratio)
                train_size = current_total_samples - val_size

                if train_size < actual_batch_size_stg1 or val_size < actual_batch_size_stg1 or train_size <=0 or val_size <=0 :
                    print(f"ERROR: STG1: Dataset too small or batch size non-positive. Train: {train_size}, Val: {val_size}. Min {actual_batch_size_stg1} needed for each. Skipping STG1.", flush=True)
                else:
                    stg1_train_subset, stg1_val_subset = random_split(full_stg1_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(SEED))
                    print(f"INFO: STG1: Dataset ({len(full_stg1_dataset)}) -> Train: {len(stg1_train_subset)}, Val: {len(stg1_val_subset)}", flush=True)
                    stg1_train_loader = DataLoader(stg1_train_subset, actual_batch_size_stg1, shuffle=True, num_workers=NUM_DATALOADER_WORKERS, pin_memory=True, drop_last=True)
                    # Ensure val_loader can be empty if val_size is 0 or too small
                    stg1_val_loader = DataLoader(stg1_val_subset, actual_batch_size_stg1, shuffle=False, num_workers=NUM_DATALOADER_WORKERS, pin_memory=True, drop_last=False) if val_size > 0 else None


                    if len(stg1_train_loader) == 0 : print(f"ERROR STG1: Train loader empty. Skipping.", flush=True)
                    else:
                        encoder = ResNetEncoder(CONTRASTIVE_FINAL_EMBEDDING_DIM, CONTRASTIVE_RESNET_VARIANT, RESNET_INPUT_CHANNELS).to(DEVICE)
                        proj_head = ProjectionHead(CONTRASTIVE_FINAL_EMBEDDING_DIM, CONTRASTIVE_PROJECTION_HEAD_HIDDEN_DIM, CONTRASTIVE_PROJECTION_DIM).to(DEVICE)
                        contrastive_net = ContrastiveTripletNetwork(encoder, proj_head).to(DEVICE)
                        criterion_stg1 = TripletNTXentLoss(CONTRASTIVE_TEMPERATURE, DEVICE)
                        optimizer_stg1 = optim.Adam(contrastive_net.parameters(), lr=CONTRASTIVE_LEARNING_RATE, weight_decay=1e-5)
                        scheduler_stg1 = get_simclr_scheduler(optimizer_stg1, CONTRASTIVE_WARMUP_EPOCHS, CONTRASTIVE_NUM_EPOCHS, CONTRASTIVE_LEARNING_RATE)
                        
                        train_contrastive_network(contrastive_net, stg1_train_loader, stg1_val_loader, criterion_stg1, optimizer_stg1, scheduler_stg1,
                                                  CONTRASTIVE_NUM_EPOCHS, DEVICE, stg1_output_dir,
                                                  stg1_model_base_filename, stg1_losses_basename,
                                                  (RESNET_INPUT_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH), # For tracing the encoder
                                                  CONTRASTIVE_EARLY_STOPPING_PATIENCE, CONTRASTIVE_MIN_DELTA,
                                                  gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS)
                        encoder_path_from_stage1_run = os.path.join(stg1_output_dir, f"{stg1_model_base_filename}_final.pth")
            else: print("WARNING: STG1: Contrastive dataset empty. Skipping pre-training.", flush=True)
        except Exception as e: print(f"CRITICAL ERROR: STG1 Failed: {e}\n{traceback.format_exc()}", flush=True)
        finally:
            if full_stg1_dataset and hasattr(full_stg1_dataset, 'close'): full_stg1_dataset.close()
    else: print("\n--- Skipping Stage 1: Contrastive Pre-training (CONTRASTIVE_NUM_EPOCHS=0) ---", flush=True)

    # Note: The ResNet encoder loading logic (PRETRAINED_ENCODER_PATH_FOR_STAGE2) is kept here
    # in case another part of the pipeline or a different type of classifier might use it.
    # However, the SimpleCNN classifier below does NOT use this `loaded_encoder`.
    loaded_encoder = None # This is the ResNet encoder, not used by the SimpleCNN below
    encoder_path_to_load_resnet = None
    if CONTRASTIVE_NUM_EPOCHS > 0:
        if encoder_path_from_stage1_run and os.path.exists(encoder_path_from_stage1_run):
            encoder_path_to_load_resnet = encoder_path_from_stage1_run
        # else: message about missing encoder from Stage 1 run already handled or implied
    elif PRETRAINED_ENCODER_PATH_FOR_STAGE2 and os.path.exists(PRETRAINED_ENCODER_PATH_FOR_STAGE2):
        encoder_path_to_load_resnet = PRETRAINED_ENCODER_PATH_FOR_STAGE2

    if encoder_path_to_load_resnet:
        print(f"INFO: Pre-trained ResNet encoder available at: {encoder_path_to_load_resnet} (Note: Not used by the current SimpleCNN classifier).", flush=True)
        # Optionally load it if needed for other purposes:
        # loaded_encoder = ResNetEncoder(...)
        # loaded_encoder.load_state_dict(...)
    else:
        print("INFO: No pre-trained ResNet encoder specified or found from Stage 1. (Note: Not used by the current SimpleCNN classifier).", flush=True)


    # Stage 2: Classifier Training (SimpleCNN directly on raw+reco images)
    if BKG_ISOLATION_CLASSIFIER_NUM_EPOCHS > 0:
        if len(MUON_NEUTRINO_CC_BACKGROUND_CATEGORIES) < 2:
            print("WARNING: Need at least two distinct background categories for isolation classifiers. Skipping Stage 2.", flush=True)
        else:
            print(f"\n--- Stage 2: Training Background Isolation Classifiers (SimpleCNN on raw+reco images) ---", flush=True)
            all_mu_nu_cc_bkg_set = set(MUON_NEUTRINO_CC_BACKGROUND_CATEGORIES)
            for target_bkg_cat in MUON_NEUTRINO_CC_BACKGROUND_CATEGORIES:
                stg2_run_ts = get_timestamp()
                classifier_name_base = f"BkgIsoDirectImgClassifier_Cat{target_bkg_cat}" # New name
                stg2_output_dir = os.path.join(BASE_MODEL_OUTPUT_DIR, f"S2_{classifier_name_base}_{stg2_run_ts}")
                os.makedirs(stg2_output_dir, exist_ok=True)
                
                stg2_losses_basename = f"direct_img_classifier_run_metrics_Cat{target_bkg_cat}"
                print(f"\nINFO: Preparing {classifier_name_base} (Run ID: {stg2_run_ts})", flush=True)

                positive_categories = [target_bkg_cat]
                negative_categories = list(all_mu_nu_cc_bkg_set - set(positive_categories))
                if not negative_categories:
                    print(f"WARNING: No negative categories for {classifier_name_base}. Skipping this classifier.", flush=True); continue

                full_stg2_dataset = None
                try:
                    # MODIFIED Instantiation: No encoder, no device for dataset
                    full_stg2_dataset = RootBinaryClassifierDataset(
                        ROOT_FILE_PATH, TREE_NAME,
                        RAW_IMAGE_BRANCH_NAMES, RECO_IMAGE_BRANCH_NAMES, EVENT_CATEGORY_BRANCH_NAME,
                        positive_categories, negative_categories,
                        IMAGE_HEIGHT, IMAGE_WIDTH, # Dataset needs H, W to process images
                        dataset_name=classifier_name_base,
                        signal_categories_to_avoid_in_dataset=SIGNAL_CATEGORIES_TO_EXCLUDE_ENTIRELY
                    )
                    if len(full_stg2_dataset) > 0:
                        val_ratio_cls = 0.2
                        actual_batch_size_stg2 = BKG_ISOLATION_CLASSIFIER_BATCH_SIZE if BKG_ISOLATION_CLASSIFIER_BATCH_SIZE > 0 else 1
                        min_val_samples_cls = actual_batch_size_stg2 * 2
                        
                        current_total_samples_cls = len(full_stg2_dataset)
                        val_size_cls = 0
                        if current_total_samples_cls * val_ratio_cls < min_val_samples_cls and current_total_samples_cls > actual_batch_size_stg2 * 4:
                            val_size_cls = min_val_samples_cls
                        elif current_total_samples_cls * val_ratio_cls >= min_val_samples_cls:
                            val_size_cls = int(current_total_samples_cls * val_ratio_cls)
                        
                        train_size_cls = current_total_samples_cls - val_size_cls
                        
                        if train_size_cls < actual_batch_size_stg2 or val_size_cls < actual_batch_size_stg2 or train_size_cls <=0 or val_size_cls <=0:
                            print(f"WARNING: {classifier_name_base}: Dataset too small or batch size non-positive. Train: {train_size_cls}, Val: {val_size_cls}. Skipping.", flush=True)
                            if full_stg2_dataset: full_stg2_dataset.close(); continue
                        
                        stg2_train_subset, stg2_val_subset = random_split(full_stg2_dataset, [train_size_cls, val_size_cls], generator=torch.Generator().manual_seed(SEED))
                        stg2_train_loader = DataLoader(stg2_train_subset, actual_batch_size_stg2, shuffle=True, num_workers=NUM_DATALOADER_WORKERS, pin_memory=True, drop_last=True)
                        stg2_val_loader = DataLoader(stg2_val_subset, actual_batch_size_stg2, shuffle=False, num_workers=NUM_DATALOADER_WORKERS, pin_memory=True, drop_last=False) if val_size_cls > 0 else None


                        if len(stg2_train_loader) == 0:
                            print(f"WARNING {classifier_name_base}: Train loader empty. Skipping.", flush=True)
                            if full_stg2_dataset: full_stg2_dataset.close(); continue

                        # MODIFIED SimpleCNN instantiation
                        bkg_cnn = SimpleCNN(input_channels=RESNET_INPUT_CHANNELS, # Should be 2
                                            map_h=IMAGE_HEIGHT,
                                            map_w=IMAGE_WIDTH).to(DEVICE)
                        criterion_cnn = nn.BCEWithLogitsLoss()
                        optimizer_cnn = optim.Adam(bkg_cnn.parameters(), lr=BKG_ISOLATION_CLASSIFIER_LEARNING_RATE)
                        scheduler_cnn = ReduceLROnPlateau(optimizer_cnn, mode='min', factor=CLASSIFIER_LR_SCHEDULER_FACTOR, patience=CLASSIFIER_LR_SCHEDULER_PATIENCE, min_lr=1e-7, verbose=True)

                        # MODIFIED cnn_input_shape_for_trace
                        train_binary_classifier(classifier_name_base, bkg_cnn, stg2_train_loader, stg2_val_loader, criterion_cnn, optimizer_cnn, scheduler_cnn,
                                                BKG_ISOLATION_CLASSIFIER_NUM_EPOCHS, DEVICE, stg2_output_dir,
                                                classifier_name_base, 
                                                stg2_losses_basename, 
                                                (RESNET_INPUT_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH), # Shape for JIT trace
                                                CLASSIFIER_EARLY_STOPPING_PATIENCE, CLASSIFIER_MIN_DELTA,
                                                print_val_batch_metrics=PRINT_VALIDATION_BATCH_METRICS)
                    else: print(f"WARNING: {classifier_name_base} dataset empty. Skipping.", flush=True)
                except Exception as e: print(f"CRITICAL ERROR: Stage 2 ({classifier_name_base}): {e}\n{traceback.format_exc()}", flush=True)
                finally:
                    if full_stg2_dataset and hasattr(full_stg2_dataset, 'close'): full_stg2_dataset.close()
    elif BKG_ISOLATION_CLASSIFIER_NUM_EPOCHS <= 0:
        print("\n--- Skipping Stage 2: Background Isolation Classifiers (BKG_ISOLATION_CLASSIFIER_NUM_EPOCHS is set to 0 or less) ---", flush=True)
    # No 'else' needed here as dependency on loaded_encoder is removed for this specific classifier.

    print(f"\nScript finished. Main run start timestamp was: {main_run_timestamp}", flush=True)