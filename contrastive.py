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

ROOT_FILE_PATH = "filtered_data.root"
TREE_NAME = "numi_fhc_overlay_inclusive_genie_run1_run1"
MODEL_OUTPUT_DIR = "trained_bkg_models"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_DATALOADER_WORKERS = 0
SEED = 42

RAW_IMAGE_BRANCH_NAMES = ["raw_image_u", "raw_image_v", "raw_image_w"]
RECO_IMAGE_BRANCH_NAMES = ["reco_image_u", "reco_image_v", "reco_image_w"]
ALL_IMAGE_BRANCH_LISTS = [RAW_IMAGE_BRANCH_NAMES, RECO_IMAGE_BRANCH_NAMES] 
CLASSIFIER_TARGET_PLANE_BRANCH_NAMES = RAW_IMAGE_BRANCH_NAMES 
EVENT_CATEGORY_BRANCH_NAME = "event_category"

IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512
RESNET_INPUT_CHANNELS = 2 

MUON_NEUTRINO_CC_BACKGROUND_CATEGORIES = [20, 21, 100, 101, 102, 103]
CONTRASTIVE_PRETRAINING_CATEGORIES = MUON_NEUTRINO_CC_BACKGROUND_CATEGORIES
SIGNAL_CATEGORIES_TO_EXCLUDE_ENTIRELY = [0, 1, 2, 10, 11]

CONTRASTIVE_RESNET_VARIANT = 'resnet18' 
CONTRASTIVE_FINAL_EMBEDDING_DIM = 128 
CONTRASTIVE_PROJECTION_HEAD_HIDDEN_DIM = 512 
CONTRASTIVE_PROJECTION_DIM = 128 
CONTRASTIVE_LEARNING_RATE = 3e-4
CONTRASTIVE_BATCH_SIZE = 64 
CONTRASTIVE_NUM_EPOCHS = 15
CONTRASTIVE_WARMUP_EPOCHS = 3
CONTRASTIVE_TEMPERATURE = 0.07
ENCODER_STATE_DICT_FILENAME_FINAL = "bkg_resnet_encoder_final.pth"
ENCODER_TORCHSCRIPT_FILENAME_FINAL = "bkg_resnet_encoder_final.pt"
ENCODER_STATE_DICT_SAVE_PATH_FINAL = os.path.join(MODEL_OUTPUT_DIR, ENCODER_STATE_DICT_FILENAME_FINAL)
ENCODER_TORCHSCRIPT_SAVE_PATH_FINAL = os.path.join(MODEL_OUTPUT_DIR, ENCODER_TORCHSCRIPT_FILENAME_FINAL)

CLASSIFIER_ENCODER_OUTPUT_DIM_FOR_CNN_INPUT = CONTRASTIVE_FINAL_EMBEDDING_DIM
CLASSIFIER_CNN_INPUT_CHANNELS = 1 
_sqrt_dim = int(math.sqrt(CLASSIFIER_ENCODER_OUTPUT_DIM_FOR_CNN_INPUT))
if _sqrt_dim * _sqrt_dim == CLASSIFIER_ENCODER_OUTPUT_DIM_FOR_CNN_INPUT:
    CLASSIFIER_CNN_FEATURE_MAP_SIZE = (_sqrt_dim, _sqrt_dim)
else:
    CLASSIFIER_CNN_FEATURE_MAP_SIZE = (CLASSIFIER_ENCODER_OUTPUT_DIM_FOR_CNN_INPUT, 1)

BKG_ISOLATION_CLASSIFIER_LEARNING_RATE = 1e-3
BKG_ISOLATION_CLASSIFIER_BATCH_SIZE = 64 
BKG_ISOLATION_CLASSIFIER_NUM_EPOCHS = 10

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

class RootContrastiveDataset(Dataset):
    def __init__(self, root_file_path, tree_name, raw_branch_names, reco_branch_names,
                 event_category_branch, image_height, image_width,
                 categories_to_use_for_training, signal_categories_to_avoid):
        super().__init__()
        self.root_file_path = root_file_path; self.tree_name = tree_name
        self.raw_branch_names = raw_branch_names; self.reco_branch_names = reco_branch_names
        self.num_plane_types = len(raw_branch_names)
        self.event_category_branch = event_category_branch
        self.image_height = image_height; self.image_width = image_width
        self.categories_to_use_for_training = set(categories_to_use_for_training)
        self.signal_categories_to_avoid = set(signal_categories_to_avoid)
        self.event_plane_pair_indices = []; self.root_file_handle = None
        try: self._open_root_file(); self._prepare_indices()
        except Exception as e: print(f"CRITICAL ERROR: ContrastiveDataset Init Error: {e}", flush=True); self._close_root_file(); raise
    def _open_root_file(self):
        if not self.root_file_handle: self.root_file_handle = uproot.open(self.root_file_path); self.tree = self.root_file_handle[self.tree_name]
    def _close_root_file(self):
        if self.root_file_handle: self.root_file_handle.close(); self.root_file_handle = None
    def close(self): self._close_root_file()
    def _prepare_indices(self):
        print(f"INFO: ContrastiveDataset: Preparing combined raw+reco inter-plane pairs. Using: {self.categories_to_use_for_training}. Avoiding: {self.signal_categories_to_avoid}", flush=True)
        event_categories_data = self.tree[self.event_category_branch].array(library="ak"); num_events = len(event_categories_data)
        for event_idx in range(num_events):
            event_cat = event_categories_data[event_idx]
            if event_cat in self.signal_categories_to_avoid or event_cat not in self.categories_to_use_for_training: continue
            for p1_idx in range(self.num_plane_types):
                for p2_idx in range(p1_idx + 1, self.num_plane_types): self.event_plane_pair_indices.append((event_idx, p1_idx, p2_idx))
        print(f"INFO: ContrastiveDataset (Full): Prepared {len(self.event_plane_pair_indices)} inter-plane pair definitions for combined raw+reco inputs.", flush=True)
    def __len__(self): return len(self.event_plane_pair_indices)
    def __getitem__(self, idx):
        if not self.root_file_handle: self._open_root_file()
        event_idx, plane_idx1, plane_idx2 = self.event_plane_pair_indices[idx]
        raw_branch1 = self.raw_branch_names[plane_idx1]; reco_branch1 = self.reco_branch_names[plane_idx1]
        raw_branch2 = self.raw_branch_names[plane_idx2]; reco_branch2 = self.reco_branch_names[plane_idx2]
        branches_to_fetch = [raw_branch1, reco_branch1, raw_branch2, reco_branch2]
        try:
            event_data = self.tree.arrays(branches_to_fetch, entry_start=event_idx, entry_stop=event_idx + 1, library="ak")
            raw_img1 = process_single_plane_image_data(ak.to_numpy(event_data[raw_branch1][0]), self.image_height, self.image_width)
            reco_img1 = process_single_plane_image_data(ak.to_numpy(event_data[reco_branch1][0]), self.image_height, self.image_width)
            view1_tensor = torch.cat((raw_img1, reco_img1), dim=0)
            raw_img2 = process_single_plane_image_data(ak.to_numpy(event_data[raw_branch2][0]), self.image_height, self.image_width)
            reco_img2 = process_single_plane_image_data(ak.to_numpy(event_data[reco_branch2][0]), self.image_height, self.image_width)
            view2_tensor = torch.cat((raw_img2, reco_img2), dim=0)
            return view1_tensor, view2_tensor
        except Exception as e: print(f"ERROR: ContrastiveDataset: GetItem Error for entry {idx} (event {event_idx}, planes {plane_idx1},{plane_idx2}): {e}", flush=True); return torch.empty(0,0), torch.empty(0,0)

class RootBinaryClassifierDataset(Dataset):
    def __init__(self, root_file_path, tree_name, raw_image_branch_names_ordered, reco_image_branch_names_ordered, event_category_branch,
                 encoder_model, positive_event_categories, negative_event_categories, image_height, image_width, 
                 cnn_input_channels_for_classifier_cnn, cnn_feature_map_size_for_classifier_cnn, device, dataset_name="Classifier", signal_categories_to_avoid_in_dataset=None):
        super().__init__(); self.root_file_path, self.tree_name = root_file_path, tree_name
        self.raw_image_branch_names = raw_image_branch_names_ordered; self.reco_image_branch_names = reco_image_branch_names_ordered
        self.num_plane_types = len(raw_image_branch_names_ordered); self.event_category_branch = event_category_branch
        self.encoder_model = encoder_model.to(device).eval(); self.image_height = image_height; self.image_width = image_width
        self.positive_categories = set(positive_event_categories); self.negative_categories = set(negative_event_categories)
        self.signal_categories_to_avoid_in_dataset = set(signal_categories_to_avoid_in_dataset) if signal_categories_to_avoid_in_dataset else set()
        self.cnn_input_channels_for_classifier_cnn = cnn_input_channels_for_classifier_cnn; self.cnn_feature_map_size_for_classifier_cnn = cnn_feature_map_size_for_classifier_cnn
        self.device, self.dataset_name = device, dataset_name; self.event_plane_info_list = []; self.root_file_handle = None
        try: self._open_root_file(); self._prepare_indices()
        except Exception as e: print(f"CRITICAL ERROR: {self.dataset_name} Dataset: Init Error: {e}", flush=True); self._close_root_file(); raise
    def _open_root_file(self):
        if not self.root_file_handle: self.root_file_handle = uproot.open(self.root_file_path); self.tree = self.root_file_handle[self.tree_name]
    def _close_root_file(self):
        if self.root_file_handle: self.root_file_handle.close(); self.root_file_handle = None
    def close(self): self._close_root_file()
    def _prepare_indices(self):
        print(f"INFO: {self.dataset_name} Dataset (Full): Preparing. Positive: {self.positive_categories}, Negative: {self.negative_categories}, Avoiding: {self.signal_categories_to_avoid_in_dataset}", flush=True)
        event_categories_data = self.tree[self.event_category_branch].array(library="ak"); num_pos_events, num_neg_events = 0,0; unique_events_processed = set()
        for event_idx, event_cat_val in enumerate(event_categories_data):
            if event_cat_val in self.signal_categories_to_avoid_in_dataset: continue
            label = -1; is_pos = event_cat_val in self.positive_categories; is_neg = event_cat_val in self.negative_categories
            if is_pos: label = 1
            elif is_neg: label = 0
            else: continue
            if event_idx not in unique_events_processed:
                if is_pos: num_pos_events +=1
                if is_neg: num_neg_events +=1
                unique_events_processed.add(event_idx)
            for plane_type_idx in range(self.num_plane_types): self.event_plane_info_list.append((event_idx, plane_type_idx, label)) 
        print(f"INFO: {self.dataset_name} Dataset (Full): Prepared {len(self.event_plane_info_list)} items. Unique Positive Events: {num_pos_events}, Unique Negative Events: {num_neg_events}.", flush=True)
    def __len__(self): return len(self.event_plane_info_list)
    def __getitem__(self, idx):
        if not self.root_file_handle: self._open_root_file()
        actual_event_idx, plane_type_idx, label = self.event_plane_info_list[idx]
        raw_branch_name = self.raw_image_branch_names[plane_type_idx]; reco_branch_name = self.reco_image_branch_names[plane_type_idx]
        try:
            event_data = self.tree.arrays([raw_branch_name, reco_branch_name], entry_start=actual_event_idx, entry_stop=actual_event_idx + 1, library="ak")
            raw_img = process_single_plane_image_data(ak.to_numpy(event_data[raw_branch_name][0]), self.image_height, self.image_width)
            reco_img = process_single_plane_image_data(ak.to_numpy(event_data[reco_branch_name][0]), self.image_height, self.image_width)
            combined_input_tensor = torch.cat((raw_img, reco_img), dim=0).unsqueeze(0).to(self.device)
            with torch.no_grad(): embedding = self.encoder_model(combined_input_tensor).squeeze(0)
            reshaped_embedding = embedding.reshape(self.cnn_input_channels_for_classifier_cnn, *self.cnn_feature_map_size_for_classifier_cnn)
            label_tensor = torch.tensor(label, dtype=torch.float32)
            return reshaped_embedding.cpu(), label_tensor.cpu()
        except Exception as e: print(f"ERROR: {self.dataset_name} Dataset: GetItem Error idx {idx} (event {actual_event_idx}, plane {plane_type_idx}): {e}", flush=True); return torch.empty(0), torch.empty(0)

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

class ContrastiveSiameseNetwork(nn.Module):
    def __init__(self, encoder, projection_head):
        super().__init__(); self.encoder = encoder; self.projection_head = projection_head
    def forward(self, view1, view2): emb1 = self.encoder(view1); emb2 = self.encoder(view2); return self.projection_head(emb1), self.projection_head(emb2)

class NTXentLoss(nn.Module):
    def __init__(self, temperature, device):
        super().__init__(); self.temperature = temperature; self.device = device
        self.criterion = nn.CrossEntropyLoss(reduction="sum"); self.similarity_function = nn.CosineSimilarity(dim=2)
    def forward(self, proj_i, proj_j):
        batch_size = proj_i.shape[0]; representations = torch.cat([proj_i, proj_j], dim=0)
        sim_matrix = self.similarity_function(representations.unsqueeze(1), representations.unsqueeze(0))
        mask_self_similarity = torch.eye(2 * batch_size, dtype=torch.bool, device=self.device)
        sim_matrix = sim_matrix.masked_fill(mask_self_similarity, -float('inf')); sim_matrix = sim_matrix / self.temperature
        labels_row1 = torch.arange(batch_size, 2 * batch_size, device=self.device); labels_row2 = torch.arange(batch_size, device=self.device)
        labels = torch.cat([labels_row1, labels_row2]).long(); loss = self.criterion(sim_matrix, labels); return loss / (2 * batch_size)

class SimpleCNN(nn.Module):
    def __init__(self, input_channels, map_h, map_w, num_classes=1):
        super().__init__(); self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, padding=1); self.bn1 = nn.BatchNorm2d(16); self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2); pooled_height = map_h // 2; pooled_width = map_w // 2
        pooled_height = max(1, pooled_height); pooled_width = max(1, pooled_width)
        self.flat_size = 16 * pooled_height * pooled_width; self.fc1 = nn.Linear(self.flat_size, num_classes)
    def forward(self, x): x = self.pool1(self.relu1(self.bn1(self.conv1(x)))); x = x.view(-1, self.flat_size); x = self.fc1(x); return x

def get_simclr_scheduler(optimizer, warmup_epochs, total_epochs, initial_lr_base):
    warmup_epochs = max(1, warmup_epochs); total_epochs = max(total_epochs, warmup_epochs)
    def lr_lambda(current_epoch):
        if current_epoch < warmup_epochs: return float(current_epoch + 1) / float(warmup_epochs) 
        else: progress = float(current_epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs)); return 0.5 * (1.0 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)

def train_contrastive_network(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs, device, model_dir, final_state_dict_path, final_torchscript_path, batch_losses_save_path, example_input_shape_for_trace):
    print(f"INFO: Starting Background SimCLR-style Pre-training with ResNet and Combined Inputs...", flush=True)
    all_train_batch_losses = []; all_val_batch_losses = []
    amp_enabled = (device.type == 'cuda')
    scaler = GradScaler(enabled=amp_enabled)
    
    for epoch in range(epochs):
        epoch_train_losses_sum = 0.0; num_train_batches = 0; model.train() 
        val_iter = iter(val_loader) if val_loader and len(val_loader) > 0 else None
        for batch_idx, (view1_train, view2_train) in enumerate(train_loader):
            current_train_loss_val = float('nan'); current_val_loss_val = float('nan')
            if view1_train.nelement() == 0 or view2_train.nelement() == 0: print(f"WARNING: Contrastive Epoch {epoch+1} Train Batch {batch_idx+1}: Empty batch.", flush=True); continue
            view1_train, view2_train = view1_train.to(device), view2_train.to(device)
            optimizer.zero_grad(); model.train()
            
            autocast_context = torch.cuda.amp.autocast() if amp_enabled else contextlib.nullcontext()

            with autocast_context:
                proj_view1_train, proj_view2_train = model(view1_train, view2_train)
                train_loss = criterion(proj_view1_train, proj_view2_train)
            current_train_loss_val = train_loss.item(); all_train_batch_losses.append(current_train_loss_val)
            epoch_train_losses_sum += current_train_loss_val; num_train_batches +=1
            
            if val_iter:
                model.eval()
                with torch.no_grad(), autocast_context:
                    try: view1_val, view2_val = next(val_iter)
                    except StopIteration: val_iter = iter(val_loader); view1_val, view2_val = next(val_iter, (None,None))
                    if view1_val is not None and view1_val.nelement() > 0 and view2_val is not None and view2_val.nelement() > 0 :
                        view1_val, view2_val = view1_val.to(device), view2_val.to(device)
                        proj_view1_val, proj_view2_val = model(view1_val, view2_val)
                        val_loss = criterion(proj_view1_val, proj_view2_val); current_val_loss_val = val_loss.item()
                all_val_batch_losses.append(current_val_loss_val)
            else: all_val_batch_losses.append(float('nan'))
            
            print(f"INFO: Contrastive Epoch [{epoch+1}/{epochs}] Train Batch [{batch_idx+1}/{len(train_loader)}] Loss: {current_train_loss_val:.4f} Val Loss: {current_val_loss_val:.4f} LR: {optimizer.param_groups[0]['lr']:.6e}", flush=True)
            
            model.train()
            if amp_enabled: scaler.scale(train_loss).backward(); scaler.step(optimizer); scaler.update()
            else: train_loss.backward(); optimizer.step()
            
        if scheduler: scheduler.step()
        avg_epoch_train_loss = epoch_train_losses_sum / num_train_batches if num_train_batches > 0 else 0
        print(f"INFO: Contrastive Epoch [{epoch+1}/{epochs}] Average Training Loss: {avg_epoch_train_loss:.4f}", flush=True)
        epoch_state_dict_path = os.path.join(model_dir, f"bkg_resnet_encoder_epoch_{epoch+1}.pth"); epoch_torchscript_path = os.path.join(model_dir, f"bkg_resnet_encoder_epoch_{epoch+1}.pt")
        torch.save(model.encoder.state_dict(), epoch_state_dict_path)
        try:
            model.encoder.eval(); example_input = torch.randn(1, *example_input_shape_for_trace, device=device)
            traced_encoder_epoch = torch.jit.trace(model.encoder, example_input); traced_encoder_epoch.save(epoch_torchscript_path)
            model.train(); print(f"INFO: Saved TorchScript ResNet encoder for epoch {epoch+1} to {epoch_torchscript_path}", flush=True)
        except Exception as e: print(f"ERROR: Failed to save TorchScript ResNet encoder for epoch {epoch+1}: {e}", flush=True); model.train() 
    torch.save(model.encoder.state_dict(), final_state_dict_path)
    print(f"INFO: Contrastive final ResNet encoder state_dict saved to {final_state_dict_path}", flush=True)
    try:
        model.encoder.eval(); example_input = torch.randn(1, *example_input_shape_for_trace, device=device)
        traced_encoder = torch.jit.trace(model.encoder, example_input); traced_encoder.save(final_torchscript_path)
        print(f"INFO: Contrastive final ResNet encoder TorchScript model saved to {final_torchscript_path}", flush=True)
    except Exception as e: print(f"ERROR: Failed to save Final ResNet Encoder TorchScript model: {e}", flush=True)
    np.savez(batch_losses_save_path, train_batch_losses=np.array(all_train_batch_losses), val_batch_losses=np.array(all_val_batch_losses))
    print(f"INFO: Contrastive train/val batch losses saved to {batch_losses_save_path}", flush=True)

def train_binary_classifier(classifier_name, cnn_model, train_loader, val_loader, criterion, optimizer, scheduler, epochs, device, model_dir, final_state_dict_path, final_torchscript_path, batch_losses_save_path, cnn_input_shape_for_trace):
    print(f"INFO: Starting training for {classifier_name}...", flush=True)
    all_train_batch_losses = []; all_val_batch_losses = []
    amp_enabled = (device.type == 'cuda')
    scaler = GradScaler(enabled=amp_enabled)
    for epoch in range(epochs):
        epoch_train_losses_sum = 0.0; num_train_batches = 0; epoch_correct_preds = 0; epoch_total_samples = 0
        cnn_model.train(); val_iter = iter(val_loader) if val_loader and len(val_loader) > 0 else None
        for batch_idx, (features_train, labels_train) in enumerate(train_loader):
            current_train_loss_val = float('nan'); current_val_loss_val = float('nan')
            if features_train.nelement() == 0: print(f"WARNING: {classifier_name} Epoch {epoch+1} Train Batch {batch_idx+1}: Empty batch skipped.", flush=True); continue
            features_train, labels_train = features_train.to(device), labels_train.to(device).unsqueeze(1)
            optimizer.zero_grad(); cnn_model.train()
            autocast_context = torch.cuda.amp.autocast() if amp_enabled else contextlib.nullcontext()
            with autocast_context:
                outputs_train = cnn_model(features_train); train_loss = criterion(outputs_train, labels_train)
            current_train_loss_val = train_loss.item(); all_train_batch_losses.append(current_train_loss_val)
            epoch_train_losses_sum += current_train_loss_val; num_train_batches +=1
            preds = torch.sigmoid(outputs_train.float()) > 0.5; epoch_correct_preds += (preds == labels_train.bool()).sum().item(); epoch_total_samples += labels_train.size(0)
            if val_iter:
                cnn_model.eval()
                with torch.no_grad(), autocast_context:
                    try: features_val, labels_val = next(val_iter)
                    except StopIteration: val_iter = iter(val_loader); features_val, labels_val = next(val_iter, (None,None))
                    if features_val is not None and features_val.nelement() > 0:
                        features_val, labels_val = features_val.to(device), labels_val.to(device).unsqueeze(1)
                        outputs_val = cnn_model(features_val); val_loss = criterion(outputs_val, labels_val); current_val_loss_val = val_loss.item()
                all_val_batch_losses.append(current_val_loss_val)
            else: all_val_batch_losses.append(float('nan'))
            print(f"INFO: {classifier_name} Epoch [{epoch+1}/{epochs}] Train Batch [{batch_idx+1}/{len(train_loader)}] Loss: {current_train_loss_val:.4f} Val Loss: {current_val_loss_val:.4f} LR: {optimizer.param_groups[0]['lr']:.6e}", flush=True)
            cnn_model.train()
            if amp_enabled: scaler.scale(train_loss).backward(); scaler.step(optimizer); scaler.update()
            else: train_loss.backward(); optimizer.step()
        if scheduler: scheduler.step()
        avg_epoch_train_loss = epoch_train_losses_sum / num_train_batches if num_train_batches > 0 else 0
        epoch_accuracy = epoch_correct_preds / epoch_total_samples if epoch_total_samples > 0 else 0
        print(f"INFO: {classifier_name} Epoch [{epoch+1}/{epochs}] Avg Training Loss: {avg_epoch_train_loss:.4f}, Training Accuracy: {epoch_accuracy:.4f}", flush=True)
        epoch_state_dict_path = os.path.join(model_dir, f"{classifier_name}_epoch_{epoch+1}.pth"); epoch_torchscript_path = os.path.join(model_dir, f"{classifier_name}_epoch_{epoch+1}.pt")
        torch.save(cnn_model.state_dict(), epoch_state_dict_path)
        try:
            cnn_model.eval(); example_input_cnn = torch.randn(1, *cnn_input_shape_for_trace, device=device)
            traced_cnn_epoch = torch.jit.trace(cnn_model, example_input_cnn); traced_cnn_epoch.save(epoch_torchscript_path)
            cnn_model.train(); print(f"INFO: Saved TorchScript model for {classifier_name} epoch {epoch+1} to {epoch_torchscript_path}", flush=True)
        except Exception as e: print(f"ERROR: Failed to save TorchScript model for {classifier_name} epoch {epoch+1}: {e}", flush=True); cnn_model.train()
    torch.save(cnn_model.state_dict(), final_state_dict_path)
    print(f"INFO: {classifier_name} final state_dict saved to {final_state_dict_path}", flush=True)
    try:
        cnn_model.eval(); example_input_cnn = torch.randn(1, *cnn_input_shape_for_trace, device=device)
        traced_cnn = torch.jit.trace(cnn_model, example_input_cnn); traced_cnn.save(final_torchscript_path)
        print(f"INFO: {classifier_name} final TorchScript model saved to {final_torchscript_path}", flush=True)
    except Exception as e: print(f"ERROR: Failed to save Final {classifier_name} TorchScript model: {e}", flush=True)
    np.savez(batch_losses_save_path, train_batch_losses=np.array(all_train_batch_losses), val_batch_losses=np.array(all_val_batch_losses))
    print(f"INFO: {classifier_name} train/val batch losses saved to {batch_losses_save_path}", flush=True)

if __name__ == "__main__":
    print(f"INFO: Device: {DEVICE}. ResNet Input C,H,W: {RESNET_INPUT_CHANNELS},{IMAGE_HEIGHT},{IMAGE_WIDTH}. Encoder Final Emb Dim: {CONTRASTIVE_FINAL_EMBEDDING_DIM}", flush=True)
    print(f"INFO: Classifier CNN Input Map: {CLASSIFIER_CNN_INPUT_CHANNELS}x{CLASSIFIER_CNN_FEATURE_MAP_SIZE[0]}x{CLASSIFIER_CNN_FEATURE_MAP_SIZE[1]} from {CLASSIFIER_ENCODER_OUTPUT_DIM_FOR_CNN_INPUT} dim embedding", flush=True)
    print(f"INFO: Muon-Neutrino CC Background Categories for training: {MUON_NEUTRINO_CC_BACKGROUND_CATEGORIES}", flush=True)
    print(f"INFO: Signal Categories explicitly excluded from ALL training: {SIGNAL_CATEGORIES_TO_EXCLUDE_ENTIRELY}", flush=True)
    print(f"WARNING: Current CONTRASTIVE_BATCH_SIZE is {CONTRASTIVE_BATCH_SIZE}. For SimCLR with ResNet, this is a key parameter. Also: EPOCHS ({CONTRASTIVE_NUM_EPOCHS}), LR ({CONTRASTIVE_LEARNING_RATE}), WARMUP ({CONTRASTIVE_WARMUP_EPOCHS}), TEMP ({CONTRASTIVE_TEMPERATURE}).", flush=True)

    if CONTRASTIVE_NUM_EPOCHS > 0:
        print(f"\n--- Stage 1: Contrastive ResNet Encoder Pre-training (Combined Raw+Reco Inputs, Inter-Plane Pairs) ---", flush=True)
        full_stg1_dataset = None
        try:
            full_stg1_dataset = RootContrastiveDataset( ROOT_FILE_PATH, TREE_NAME, RAW_IMAGE_BRANCH_NAMES, RECO_IMAGE_BRANCH_NAMES, EVENT_CATEGORY_BRANCH_NAME, IMAGE_HEIGHT, IMAGE_WIDTH, CONTRASTIVE_PRETRAINING_CATEGORIES, SIGNAL_CATEGORIES_TO_EXCLUDE_ENTIRELY )
            if len(full_stg1_dataset) > 0:
                train_size = len(full_stg1_dataset) // 2; val_size = len(full_stg1_dataset) - train_size
                if train_size == 0 or val_size == 0: print(f"ERROR: STG1: Dataset too small to split. Train: {train_size}, Val: {val_size}. Skipping.", flush=True)
                else:
                    stg1_train_subset, stg1_val_subset = random_split(full_stg1_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(SEED))
                    print(f"INFO: STG1: Split full dataset of {len(full_stg1_dataset)} into Train: {len(stg1_train_subset)}, Val: {len(stg1_val_subset)}", flush=True)
                    stg1_train_loader = DataLoader(stg1_train_subset, batch_size=CONTRASTIVE_BATCH_SIZE, shuffle=True, num_workers=NUM_DATALOADER_WORKERS, pin_memory=DEVICE.type=='cuda', drop_last=True)
                    stg1_val_loader = DataLoader(stg1_val_subset, batch_size=CONTRASTIVE_BATCH_SIZE, shuffle=False, num_workers=NUM_DATALOADER_WORKERS, pin_memory=DEVICE.type=='cuda', drop_last=True)
                    if len(stg1_train_loader) == 0 : print(f"ERROR: STG1: Train loader empty. Batches: {len(stg1_train_loader)}. Skipping.", flush=True)
                    else:
                        encoder = ResNetEncoder(final_embedding_dim=CONTRASTIVE_FINAL_EMBEDDING_DIM, resnet_variant=CONTRASTIVE_RESNET_VARIANT, input_channels=RESNET_INPUT_CHANNELS).to(DEVICE)
                        proj_head = ProjectionHead(CONTRASTIVE_FINAL_EMBEDDING_DIM, CONTRASTIVE_PROJECTION_HEAD_HIDDEN_DIM, CONTRASTIVE_PROJECTION_DIM).to(DEVICE)
                        contrastive_net = ContrastiveSiameseNetwork(encoder, proj_head).to(DEVICE)
                        criterion = NTXentLoss(CONTRASTIVE_TEMPERATURE, DEVICE)
                        optimizer = optim.Adam(contrastive_net.parameters(), lr=CONTRASTIVE_LEARNING_RATE, weight_decay=1e-5)
                        scheduler = get_simclr_scheduler(optimizer, CONTRASTIVE_WARMUP_EPOCHS, CONTRASTIVE_NUM_EPOCHS, CONTRASTIVE_LEARNING_RATE)
                        batch_losses_path = os.path.join(MODEL_OUTPUT_DIR, "contrastive_resnet_batch_losses.npz")
                        example_trace_input_shape = (RESNET_INPUT_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH)
                        train_contrastive_network(contrastive_net, stg1_train_loader, stg1_val_loader if len(stg1_val_loader)>0 else None, criterion, optimizer, scheduler, CONTRASTIVE_NUM_EPOCHS, DEVICE, MODEL_OUTPUT_DIR, ENCODER_STATE_DICT_SAVE_PATH_FINAL, ENCODER_TORCHSCRIPT_SAVE_PATH_FINAL, batch_losses_path, example_trace_input_shape)
            else: print("WARNING: STG1: Contrastive dataset empty. Skipping pre-training.", flush=True)
        except Exception as e: print(f"CRITICAL ERROR: STG1 Failed: {e}", flush=True)
        finally:
            if full_stg1_dataset and hasattr(full_stg1_dataset, 'close'): full_stg1_dataset.close()
    else: print("\n--- Skipping Stage 1: Contrastive Pre-training (CONTRASTIVE_NUM_EPOCHS=0) ---", flush=True)

    loaded_encoder = None
    if os.path.exists(ENCODER_STATE_DICT_SAVE_PATH_FINAL):
        print(f"INFO: Loading pre-trained background ResNet encoder from {ENCODER_STATE_DICT_SAVE_PATH_FINAL}", flush=True)
        loaded_encoder = ResNetEncoder(final_embedding_dim=CONTRASTIVE_FINAL_EMBEDDING_DIM, resnet_variant=CONTRASTIVE_RESNET_VARIANT, input_channels=RESNET_INPUT_CHANNELS)
        try:
            loaded_encoder.load_state_dict(torch.load(ENCODER_STATE_DICT_SAVE_PATH_FINAL, map_location=DEVICE))
            loaded_encoder.to(DEVICE).eval();
            for param in loaded_encoder.parameters(): param.requires_grad = False
            print("INFO: Background ResNet encoder loaded and weights frozen.", flush=True)
        except Exception as e: print(f"ERROR: Failed to load background ResNet encoder: {e}. Classifiers cannot be trained.", flush=True); loaded_encoder = None
    else: print("WARNING: Pre-trained background ResNet encoder not found. Cannot train background isolation classifiers.", flush=True)

    if loaded_encoder and BKG_ISOLATION_CLASSIFIER_NUM_EPOCHS > 0:
        if len(MUON_NEUTRINO_CC_BACKGROUND_CATEGORIES) < 2:
            print(f"WARNING: Need at least two categories in MUON_NEUTRINO_CC_BACKGROUND_CATEGORIES for isolation classifiers. Found: {len(MUON_NEUTRINO_CC_BACKGROUND_CATEGORIES)}. Skipping Stage 2.", flush=True)
        else:
            print(f"\n--- Stage 2: Training Muon-Neutrino CC Background Isolation Classifiers ---", flush=True)
            all_mu_nu_cc_bkg_set = set(MUON_NEUTRINO_CC_BACKGROUND_CATEGORIES)
            for target_bkg_cat in MUON_NEUTRINO_CC_BACKGROUND_CATEGORIES:
                classifier_name = f"BkgIsoClassifier_Cat{target_bkg_cat}"
                positive_categories = [target_bkg_cat]; negative_categories = list(all_mu_nu_cc_bkg_set - set(positive_categories))
                if not negative_categories: print(f"WARNING: Target {target_bkg_cat}, no other MuNu CC bkg categories for negative class. Skipping.", flush=True); continue
                print(f"INFO: Preparing {classifier_name} (Positive: {positive_categories} vs. Negative: {negative_categories})", flush=True)
                full_stg2_dataset = None
                try:
                    full_stg2_dataset = RootBinaryClassifierDataset( ROOT_FILE_PATH, TREE_NAME, RAW_IMAGE_BRANCH_NAMES, RECO_IMAGE_BRANCH_NAMES, EVENT_CATEGORY_BRANCH_NAME, loaded_encoder, positive_event_categories=positive_categories, negative_event_categories=negative_categories, image_height=IMAGE_HEIGHT, image_width=IMAGE_WIDTH, cnn_input_channels_for_classifier_cnn=CLASSIFIER_CNN_INPUT_CHANNELS, cnn_feature_map_size_for_classifier_cnn=CLASSIFIER_CNN_FEATURE_MAP_SIZE, device=DEVICE, dataset_name=classifier_name, signal_categories_to_avoid_in_dataset=SIGNAL_CATEGORIES_TO_EXCLUDE_ENTIRELY )
                    if len(full_stg2_dataset) > 0:
                        train_size = len(full_stg2_dataset) // 2; val_size = len(full_stg2_dataset) - train_size
                        if train_size == 0 or val_size == 0: 
                            print(f"WARNING: {classifier_name}: Dataset too small to split. Train: {train_size}, Val: {val_size}. Skipping.", flush=True)
                            if full_stg2_dataset and hasattr(full_stg2_dataset, 'close'): full_stg2_dataset.close()
                            continue
                        stg2_train_subset, stg2_val_subset = random_split(full_stg2_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(SEED))
                        print(f"INFO: {classifier_name}: Split full dataset of {len(full_stg2_dataset)} into Train: {len(stg2_train_subset)}, Val: {len(stg2_val_subset)}", flush=True)
                        stg2_train_loader = DataLoader(stg2_train_subset, batch_size=BKG_ISOLATION_CLASSIFIER_BATCH_SIZE, shuffle=True, num_workers=NUM_DATALOADER_WORKERS, pin_memory=DEVICE.type=='cuda', drop_last=True)
                        stg2_val_loader = DataLoader(stg2_val_subset, batch_size=BKG_ISOLATION_CLASSIFIER_BATCH_SIZE, shuffle=False, num_workers=NUM_DATALOADER_WORKERS, pin_memory=DEVICE.type=='cuda', drop_last=True)
                        if len(stg2_train_loader) == 0 : 
                            print(f"WARNING: {classifier_name}: Train loader empty. Skipping.", flush=True)
                            if full_stg2_dataset and hasattr(full_stg2_dataset, 'close'): full_stg2_dataset.close()
                            continue
                        bkg_cnn = SimpleCNN(CLASSIFIER_CNN_INPUT_CHANNELS, CLASSIFIER_CNN_FEATURE_MAP_SIZE[0], CLASSIFIER_CNN_FEATURE_MAP_SIZE[1]).to(DEVICE)
                        criterion = nn.BCEWithLogitsLoss(); optimizer = optim.Adam(bkg_cnn.parameters(), lr=BKG_ISOLATION_CLASSIFIER_LEARNING_RATE)
                        scheduler_cnn = get_simclr_scheduler(optimizer, 5, BKG_ISOLATION_CLASSIFIER_NUM_EPOCHS, BKG_ISOLATION_CLASSIFIER_LEARNING_RATE) if len(stg2_train_loader) > 0 else None
                        state_dict_path_final = os.path.join(MODEL_OUTPUT_DIR, f"bkg_iso_classifier_cat{target_bkg_cat}_final.pth")
                        torchscript_path_final = os.path.join(MODEL_OUTPUT_DIR, f"bkg_iso_classifier_cat{target_bkg_cat}_final.pt")
                        batch_losses_path = os.path.join(MODEL_OUTPUT_DIR, f"bkg_iso_classifier_cat{target_bkg_cat}_batch_losses.npz")
                        train_binary_classifier(classifier_name, bkg_cnn, stg2_train_loader, stg2_val_loader if len(stg2_val_loader)>0 else None, criterion, optimizer, scheduler_cnn, BKG_ISOLATION_CLASSIFIER_NUM_EPOCHS, DEVICE, MODEL_OUTPUT_DIR, state_dict_path_final, torchscript_path_final, batch_losses_path, (CLASSIFIER_CNN_INPUT_CHANNELS, *CLASSIFIER_CNN_FEATURE_MAP_SIZE))
                    else: print(f"WARNING: {classifier_name} dataset (full) empty. Skipping.", flush=True)
                except Exception as e: print(f"CRITICAL ERROR: Stage 2 ({classifier_name}): Failed for target cat {target_bkg_cat}: {e}", flush=True)
                finally:
                    if full_stg2_dataset and hasattr(full_stg2_dataset, 'close'): full_stg2_dataset.close()
    elif BKG_ISOLATION_CLASSIFIER_NUM_EPOCHS > 0:
        print("WARNING: Skipping Stage 2: No pre-trained encoder or NUM_EPOCHS is 0.", flush=True)
    else:
        print("\n--- Skipping Stage 2: Background Isolation Classifiers (BKG_ISOLATION_CLASSIFIER_NUM_EPOCHS=0) ---", flush=True)
    print("\nScript finished.", flush=True)