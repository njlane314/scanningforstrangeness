import time
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from src.config import Config
from src.dataset import Dataset
from src.models import SimCLRModel

def multi_positive_info_nce_loss(features, num_views=3, temperature=0.5):
    features = F.normalize(features, dim=1)
    batch_size = features.shape[0] // num_views
    loss = 0.0
    total_count = 0
    similarity_matrix = torch.matmul(features, features.T)
    for i in range(batch_size):
        indices = torch.arange(i * num_views, (i + 1) * num_views, device=features.device)
        for anchor in indices:
            positives = indices[indices != anchor]
            pos_sim = torch.exp(similarity_matrix[anchor, positives] / temperature).sum()
            mask = torch.ones(features.shape[0], dtype=torch.bool, device=features.device)
            mask[indices] = False
            neg_sim = torch.exp(similarity_matrix[anchor][mask] / temperature).sum()
            loss += -torch.log(pos_sim / (pos_sim + neg_sim))
            total_count += 1
    return loss / total_count

config_path = "cfg/default.yaml"
config = Config(config_path)
dataset = Dataset(config)
print(dataset.__len__())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
model = SimCLRModel(in_channels=1, feature_dim=128, projection_hidden_dim=512, projection_dim=128).to(device)
optimiser = optim.Adam(model.parameters(), lr=1e-3)

num_epochs = 1
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    num_batches = 0
    print("Starting epoch", epoch)
    for batch_idx, (images, _, _, _, _) in enumerate(train_loader):
        B, planes, H, W = images.shape
        print(f"Batch {batch_idx}: Original images shape = {images.shape}")
        if planes < 3:
            print(f"Batch {batch_idx}: Skipped because number of planes ({planes}) is less than 3")
            continue
        images = images.view(B * 3, 1, H, W).to(device)
        print(f"Batch {batch_idx}: Reshaped images shape = {images.shape}")
        
        optimiser.zero_grad()
        
        start_forward = time.time()
        projections = model(images)
        forward_time = time.time() - start_forward
        print(f"Batch {batch_idx}: Forward pass took {forward_time:.4f} seconds")
        print(f"Batch {batch_idx}: Projections shape = {projections.shape}")
        
        start_loss = time.time()
        loss = multi_positive_info_nce_loss(projections, num_views=3, temperature=0.5)
        loss_time = time.time() - start_loss
        print(f"Batch {batch_idx}: Loss computation took {loss_time:.4f} seconds")
        print(f"Batch {batch_idx}: Iteration Loss = {loss.item():.4f}")
        
        start_backward = time.time()
        loss.backward()
        backward_time = time.time() - start_backward
        print(f"Batch {batch_idx}: Backward pass took {backward_time:.4f} seconds")
        
        start_optim = time.time()
        optimiser.step()
        optim_time = time.time() - start_optim
        print(f"Batch {batch_idx}: Optimiser step took {optim_time:.4f} seconds")
        
        total_loss += loss.item()
        num_batches += 1
        print(f"Batch {batch_idx}: Cumulative average loss so far = {total_loss/num_batches:.4f}")
        
    avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
    print(f"Epoch {epoch}: Average Loss = {avg_loss:.4f}")
    
torch.save(model.encoder.state_dict(), "pretrained_encoder.config_path")
print("Pretrained encoder saved as 'pretrained_encoder.config_path'")