import argparse
import os
import sys
import uproot
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import Adam
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score
import datetime

cudnn.benchmark = True

def get_parser():
    parser = argparse.ArgumentParser(description="UResNet")
    parser.add_argument("--num-epochs", default=10, type=int)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--learning-rate", default=0.01, type=float)
    parser.add_argument("--root-file", type=str, default="/gluster/data/dune/niclane/signal/nlane_prod_strange_resample_fhc_run2_fhc_reco2_reco2_trainingimage_signal_lambdamuon_1000_ana.root")
    parser.add_argument("--img-size", default=512, type=int)
    parser.add_argument("--target-labels", type=str, default="0,1,2,4")
    parser.add_argument("--plane", type=int, choices=[0, 1, 2], required=True)
    parser.add_argument("--output-dir", type=str, default="/gluster/data/dune/niclane/checkpoints/segmentation")
    return parser

def maxpool():
    return nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

def dropout(prob):
    return nn.Dropout(prob)

def reinit_layer(layer, leak=0.0, use_kaiming_normal=True):
    if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.ConvTranspose2d):
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
    def __init__(self, args, indices=None):
        self.args = args
        self.root_file = uproot.open(args.root_file, array_cache=None, num_workers=0)
        self.tree = self.root_file["imageanalyser/ImageTree"]
        self.plane = args.plane
        self.img_size = args.img_size
        self.target_labels = [int(x) for x in args.target_labels.split(',')]
        self.foreground_labels = [lbl for lbl in self.target_labels if lbl >= 2]
        self.num_classes = len(self.foreground_labels) + 1  
        self.enum_to_model = {val: idx for idx, val in enumerate(self.foreground_labels, start=1)}
        event_types = self.tree["type"].array(library="np")
        if indices is None:
            self.indices = np.where(event_types == 0)[0]
        else:
            self.indices = indices
        self.num_events = len(self.indices)
    def __len__(self):
        return self.num_events
    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        data = self.tree.arrays(
            ["input", "truth", "run", "subrun", "event"],
            entry_start=actual_idx, entry_stop=actual_idx + 1,
            library="np"
        )
        run, subrun, event = data["run"][0], data["subrun"][0], data["event"][0]
        img = np.fromiter(data["input"][0][self.plane], dtype=np.float32).reshape(1, self.img_size, self.img_size)
        img = np.log1p(img)
        img = torch.from_numpy(img)
        truth = np.fromiter(data["truth"][0][self.plane], dtype=np.int64).reshape(self.img_size, self.img_size)
        label_map = np.zeros_like(truth, dtype=np.int64)
        for model_idx, label in enumerate(self.foreground_labels, start=1):
            label_map[truth == label] = model_idx
        label_map = torch.from_numpy(label_map)
        return img, label_map, run, subrun, event

def get_class_weights(stats):
    if np.any(stats == 0.):
        idx = np.where(stats == 0.)
        stats[idx] = 1
        weights = 1. / stats
        weights[idx] = 0
    else:
        weights = 1. / stats
    return [weight / sum(weights) for weight in weights]

def count_classes(dataset):
    if isinstance(dataset, torch.utils.data.Subset):
        num_classes = dataset.dataset.num_classes
    else:
        num_classes = dataset.num_classes
    counts = np.zeros(num_classes, dtype=np.int64)
    for _, label_map, _, _, _ in dataset:
        counts += np.bincount(label_map.flatten(), minlength=num_classes)
    return counts

def calculate_metrics(pred, target):
    pred = pred.argmax(dim=1).flatten().cpu().numpy()
    target = target.flatten().cpu().numpy()
    accuracy = (pred == target).mean()
    precision = precision_score(target, pred, average='macro', zero_division=0)
    recall = recall_score(target, pred, average='macro', zero_division=0)
    return accuracy, precision, recall

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Currently available devices: {torch.cuda.device_count()}")

    full_dataset = ImageDataset(args)
    train_size = int(0.5 * len(full_dataset))
    valid_size = len(full_dataset) - train_size
    train_dataset, valid_dataset = random_split(full_dataset, [train_size, valid_size])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)

    #print("Calculating class counts...")
    #train_stats = count_classes(train_dataset)
    #print(f"Class counts: {train_stats}")

    #print("Calculating class weights...")
    #class_weights = get_class_weights(train_stats)
    #print(f"Class weights: {class_weights}")

    class_weights = [0.0010253595104385963, 0.40767943871333523, 0.5912952017762262]

    model = UNet(in_dim=1, n_classes=full_dataset.num_classes)
    model = nn.DataParallel(model)
    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device))
    optim = Adam(model.parameters(), lr=args.learning_rate)

    model_save_dir = os.path.join(args.output_dir, "saved_models")
    os.makedirs(model_save_dir, exist_ok=True)

    train_losses = []
    valid_losses = []
    accuracies = []
    precisions = []
    recalls = []

    for epoch in range(args.num_epochs):
        model.train()
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{args.num_epochs}")):
            img, label_map, _, _, _ = batch
            img, label_map = img.to(device), label_map.to(device, dtype=torch.long)
            optim.zero_grad()
            pred = model(img)
            train_loss = loss_fn(pred, label_map)
            model.eval()
            with torch.no_grad():
                valid_iterator = iter(valid_loader)
                valid_batch = next(valid_iterator)
                valid_img, valid_label_map, _, _, _ = valid_batch
                valid_img, valid_label_map = valid_img.to(device), valid_label_map.to(device, dtype=torch.long)
                valid_pred = model(valid_img)
                valid_loss = loss_fn(valid_pred, valid_label_map).item()
                valid_losses.append(valid_loss)
                accuracy, precision, recall = calculate_metrics(valid_pred, valid_label_map)
                accuracies.append(accuracy)
                precisions.append(precision)
                recalls.append(recall)
            model.train()
            train_loss.backward()
            optim.step()
            train_losses.append(train_loss.item())

            print(f"Epoch {epoch+1}, Batch {batch_idx+1}: Train Loss = {train_loss.item():.4f}, "
                  f"Valid Loss = {valid_loss:.4f}, Accuracy = {accuracy:.4f}, "
                  f"Precision = {precision:.4f}, Recall = {recall:.4f}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics_save_path = os.path.join(args.output_dir, f"metrics_epoch_{epoch+1}_{timestamp}.npz")
        np.savez(metrics_save_path,
                 train_losses=train_losses,
                 valid_losses=valid_losses,
                 accuracies=accuracies,
                 precisions=precisions,
                 recalls=recalls)
        print(f"Metrics saved at {metrics_save_path}")

        model_save_path = os.path.join(model_save_dir, f"unet_epoch_{epoch+1}.pt")
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved at {model_save_path}")

    print("Training complete")

