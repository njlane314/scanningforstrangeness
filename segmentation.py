import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import uproot
import numpy as np
import os
from torch.utils.data import Dataset

torch.backends.cudnn.benchmark = True

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-epochs", default=10, type=int)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--learning-rate", default=0.01, type=float)
    parser.add_argument("--root-file", type=str, default="/gluster/data/dune/niclane/signal/nl_lambda_nohadrons_reco2_validation_2000_strangenessselectionfilter_100_new_analysis.root")
    parser.add_argument("--img-size", default=512, type=int)
    parser.add_argument("--plane", type=int, choices=[0, 1, 2], required=True)
    parser.add_argument("--output-dir", type=str, default="/gluster/data/dune/niclane/checkpoints/segmentation")
    return parser

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
    def __init__(self, in_dim, n_classes, depth=4, n_filters=16, drop_prob=0.1, y_range=None):
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
        data = self.tree.arrays([self.calo_key, self.reco_key, self.label_key], entry_start=actual_idx, entry_stop=actual_idx + 1, library="np")
        
        img_calo = data[self.calo_key][0].reshape(self.img_size, self.img_size)
        img_reco = data[self.reco_key][0].reshape(self.img_size, self.img_size)
        img_calo = np.log1p(img_calo)
        img_reco = np.log1p(img_reco)
        img = np.stack([img_calo, img_reco], axis=0)  
        img = torch.from_numpy(img).float()
        
        truth = data[self.label_key][0].reshape(self.img_size, self.img_size)
        truth = torch.from_numpy(truth).long()
        
        return img, truth

def calculate_class_frequencies(dataset):
    class_counts = {}
    for _, label in dataset:
        unique, counts = torch.unique(label, return_counts=True)
        for cls, count in zip(unique, counts):
            cls = cls.item()
            class_counts[cls] = class_counts.get(cls, 0) + count.item()
    return class_counts

def calculate_class_weights(class_counts):
    total_samples = sum(class_counts.values())
    n_classes = len(class_counts)
    weights = {cls: total_samples / (n_classes * count) for cls, count in class_counts.items()}
    return weights

def main():
    parser = get_parser()
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dataset = ImageDataset(args)
    
    train_size = int(0.5 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    class_counts = calculate_class_frequencies(train_dataset)
    weights = calculate_class_weights(class_counts)
    n_classes = max(class_counts.keys()) + 1
    weight_tensor = torch.zeros(n_classes)
    for cls, weight in weights.items():
        weight_tensor[cls] = weight
    weight_tensor = weight_tensor.to(device)
    
    model = UNet(in_dim=2, n_classes=n_classes, depth=4, n_filters=16, drop_prob=0.1)
    model.to(device)
    
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    step_train_losses = []
    step_valid_losses = []
    step_learning_rates = []
    
    for epoch in range(args.num_epochs):
        model.train()
        val_loader_iter = iter(val_loader)
        
        for batch_idx, (train_images, train_labels) in enumerate(train_loader):
            train_images = train_images.to(device)
            train_labels = train_labels.to(device)
            
            optimizer.zero_grad()
            train_outputs = model(train_images)
            train_loss = criterion(train_outputs, train_labels)
            
            try:
                val_images, val_labels = next(val_loader_iter)
            except StopIteration:
                val_loader_iter = iter(val_loader)
                val_images, val_labels = next(val_loader_iter)
            
            val_images = val_images.to(device)
            val_labels = val_labels.to(device)
            
            model.eval()
            with torch.no_grad():
                val_outputs = model(val_images)
                val_loss = criterion(val_outputs, val_labels)
            model.train()
            
            step_train_losses.append(train_loss.item())
            step_valid_losses.append(val_loss.item())
            step_learning_rates.append(optimizer.param_groups[0]['lr'])
            
            train_loss.backward()
            optimizer.step()
            
            print(f"Epoch [{epoch+1}/{args.num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], Train Loss: {train_loss.item():.4f}, Val Loss: {val_loss.item():.4f}")
        
        checkpoint_path = os.path.join(args.output_dir, f"unet_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")
        
        np.savez(os.path.join(args.output_dir, "losses.npz"),
                 step_train_losses=step_train_losses,
                 step_valid_losses=step_valid_losses,
                 step_learning_rates=step_learning_rates)
    
    print("Training completed.")

if __name__ == "__main__":
    main()