import torch
import os
import argparse
from tqdm import tqdm
from models.model import UNet
from models.segmentation_dataset import SegmentationDataLoader
from utils import set_seed, create_model, get_class_weights, load_model_only, accuracy, save_model

def train_model(args):
    if torch.cuda.is_available():
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
        device = torch.device('cuda:0')
    else:
        torch.set_default_tensor_type(torch.FloatTensor)
        device = torch.device('cpu')

    model_dir = f"{args.output_dir}/models/pass{args.vertex_pass}/{args.view}/"
    os.makedirs(model_dir, exist_ok=True)

    set_seed(args.seed)

    bunch = SegmentationDataLoader(args.image_path, args.view, batch_size=args.batch_size, valid_pct=0.25, device=device)
    train_stats = bunch.count_classes(args.num_classes)
    weights = get_class_weights(train_stats)

    train_losses = torch.zeros(args.n_epochs * len(bunch.train_dl), device=device)
    val_losses = torch.zeros(args.n_epochs, device=device)
    train_accs = torch.zeros(args.n_epochs * len(bunch.train_dl), device=device)
    val_accs = torch.zeros(args.n_epochs, device=device)
    model, loss_fn, optim = create_model(args.num_classes, weights, device)
    i = 0

    for e in tqdm(range(args.n_epochs), desc="Training"):
        model.train()
        for b, batch in enumerate(bunch.train_dl):
            x, y = batch
            pred = model(x)
            loss = loss_fn(pred, y)
            train_losses[i] = loss.item()
            train_accs[i] = accuracy(pred, y)
            loss.backward()
            optim.step()
            optim.zero_grad()
            i += 1

        model.eval()
        with torch.no_grad():
            val_loss, val_acc = [], []
            for x, y in bunch.valid_dl:
                pred = model(x)
                val_loss.append(loss_fn(pred, y).item())
                val_acc.append(accuracy(pred, y))
            val_losses[e] = torch.mean(torch.tensor(val_loss))
            val_accs[e] = torch.mean(torch.tensor(val_acc))

        save_model(model, x, f"{args.output_dir}/models/pass{args.vertex_pass}/{args.view}/{args.model_name}_{e}")

    torch.set_default_tensor_type(torch.FloatTensor)

def trace_and_save_model(args):
    torch.set_default_tensor_type(torch.FloatTensor)
    device = torch.device('cpu')
    model = load_model_only(f"{args.output_dir}/models/pass{args.vertex_pass}/{args.view}/{args.model_name}_19.pkl", args.num_classes, device)
    sm = torch.jit.script(model)
    sm.save(f"{args.output_dir}/models/pass{args.vertex_pass}/{args.view}/UB_ProngFinder_{args.vertex_pass}_{args.view}.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="U-Net Vertex Finder Training")
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_classes", type=int, default=4)
    parser.add_argument("--n_epochs", type=int, default=20)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--view", type=str, default="U")
    parser.add_argument("--vertex_pass", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="outputs")
    args = parser.parse_args()
    train_model(args)
    trace_and_save_model(args)
