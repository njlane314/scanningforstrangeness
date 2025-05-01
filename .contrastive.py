import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import uproot
import numpy as np
import os
from torch.utils.data import Dataset
from torchvision import models
import torch.nn.functional as F
import random

torch.backends.cudnn.benchmark = True

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-epochs", default=10, type=int)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--learning-rate", default=0.01, type=float)
    parser.add_argument("--root-file", type=str, default="/gluster/data/dune/niclane/signal/nl_lambda_nohadrons_reco2_validation_2000_strangenessselectionfilter_100_new_analysis.root")
    parser.add_argument("--img-size", default=512, type=int)
    parser.add_argument("--output-dir", type=str, default="/gluster/data/dune/niclane/checkpoints/contrastive")
    return parser

class ImageDataset(Dataset):
    def __init__(self, args):
        self.args = args
        self.root_file = uproot.open(args.root_file, array_cache=None, num_workers=0)
        self.tree = self.root_file["strangenessFilter/EventSelectionFilter"]
        self.img_size = args.img_size
        in_fiducial_data = self.tree["in_fiducial"].array(library="np")
        truth_category = self.tree["truth_category"].array(library="np")
        self.filtered_indices = np.where((in_fiducial_data == True) & (truth_category != 1))[0]
        self.plane_letters = ['u', 'v', 'w']
        self.calo_keys = [f"calo_pixels_{p}" for p in self.plane_letters]
        self.reco_keys = [f"reco_pixels_{p}" for p in self.plane_letters]

    def __len__(self):
        return len(self.filtered_indices)

    def __getitem__(self, idx):
        actual_idx = self.filtered_indices[idx]
        plane1, plane2 = random.sample(self.plane_letters, 2)
        calo_key1, reco_key1 = f"calo_pixels_{plane1}", f"reco_pixels_{plane1}"
        calo_key2, reco_key2 = f"calo_pixels_{plane2}", f"reco_pixels_{plane2}"
        data = self.tree.arrays([calo_key1, reco_key1, calo_key2, reco_key2], entry_start=actual_idx, entry_stop=actual_idx + 1, library="np")
        
        img_calo1 = data[calo_key1][0].reshape(self.img_size, self.img_size)
        img_reco1 = data[reco_key1][0].reshape(self.img_size, self.img_size)
        img1 = np.stack([img_calo1, img_reco1], axis=0)
        img1 = np.log1p(img1)
        img1 = torch.from_numpy(img1).float()

        img_calo2 = data[calo_key2][0].reshape(self.img_size, self.img_size)
        img_reco2 = data[reco_key2][0].reshape(self.img_size, self.img_size)
        img2 = np.stack([img_calo2, img_reco2], axis=0)
        img2 = np.log1p(img2)
        img2 = torch.from_numpy(img2).float()

        return img1, img2

class ResNetContrastive(nn.Module):
    def __init__(self):
        super(ResNetContrastive, self).__init__()
        self.backbone = models.resnet50(pretrained=False)
        self.backbone.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.backbone.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 128)
        )

    def forward(self, x):
        return self.backbone(x)

def nt_xent_loss(embeddings, temperature=0.5):
    B = embeddings.size(0) // 2
    embeddings = F.normalize(embeddings, dim=1)
    sim = torch.mm(embeddings, embeddings.t()) / temperature
    pos_sim = torch.cat([sim[:B, B:].diagonal(), sim[B:, :B].diagonal()], dim=0)
    sim = sim - torch.eye(2*B, device=embeddings.device) * 1e9
    loss = -pos_sim + torch.logsumexp(sim, dim=1)
    return loss.mean()

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

    model = ResNetContrastive()
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    step_train_losses = []
    step_valid_losses = []
    step_learning_rates = []

    for epoch in range(args.num_epochs):
        model.train()
        val_loader_iter = iter(val_loader)
        for batch_idx, (train_img1, train_img2) in enumerate(train_loader):
            train_images = torch.cat([train_img1, train_img2], dim=0).to(device)
            optimizer.zero_grad()
            train_embeddings = model(train_images)
            train_loss = nt_xent_loss(train_embeddings)

            try:
                val_img1, val_img2 = next(val_loader_iter)
            except StopIteration:
                val_loader_iter = iter(val_loader)
                val_img1, val_img2 = next(val_loader_iter)
            val_images = torch.cat([val_img1, val_img2], dim=0).to(device)
            model.eval()
            with torch.no_grad():
                val_embeddings = model(val_images)
                val_loss = nt_xent_loss(val_embeddings)
            model.train()

            step_train_losses.append(train_loss.item())
            step_valid_losses.append(val_loss.item())
            step_learning_rates.append(optimizer.param_groups[0]['lr'])

            train_loss.backward()
            optimizer.step()

            print(f"Epoch [{epoch+1}/{args.num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], Train Loss: {train_loss.item():.4f}, Val Loss: {val_loss.item():.4f}")

        checkpoint_path = os.path.join(args.output_dir, f"resnet50_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

        np.savez(os.path.join(args.output_dir, "losses.npz"),
                 step_train_losses=step_train_losses,
                 step_valid_losses=step_valid_losses,
                 step_learning_rates=step_learning_rates)

    print("Training completed.")

if __name__ == "__main__":
    main()