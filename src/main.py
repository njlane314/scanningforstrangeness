import torch
import os
import argparse
from tqdm import tqdm
from dataset import SegmentationDataLoader
from utils import set_seed, create_model, get_class_weights, load_model, save_model, plot_loss, visualise_predictions
import numpy as np

def train_model(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    os.makedirs(f"{args.output_dir}/models/pass{args.vertex_pass}/{args.view}/", exist_ok=True)

    set_seed(args.seed)

    bunch = SegmentationDataLoader(args.image_path, args.view, batch_size=args.batch_size, valid_pct=0.5, device=device)

    train_stats = bunch.count_classes(args.num_classes)
    weights = get_class_weights(train_stats)

    model, loss_fn, optim = create_model(args.num_classes, weights, device)
    model = model.to(device)

    #for name, param in model.named_parameters():
    #    if 'weight' in name:
    #        print(f"Weight initialization for {name}, mean: {param.mean().item()}, std: {param.std().item()}")

    metrics = {
        'train_losses': [], 
        'valid_losses': []
    }

    train_step = 0 
    valid_step = 0 
    for e in tqdm(range(args.n_epochs), desc="Training"):
        model.train()
        for batch in bunch.train_dl:
            x, y = batch
            x, y = x.to(device), y.to(device)

            print(f"Input data range: {x.min().item()} to {x.max().item()}")
            print(f"Label data range: {y.min().item()} to {y.max().item()}")

            y = y.to(torch.long)

            optim.zero_grad()

            pred = model(x)
            #print(f"Prediction shape: {pred.shape}, Ground truth shape: {y.shape}")
            
            loss = loss_fn(pred, y)

            loss.backward()
            optim.step()

            #for name, param in model.named_parameters():
            #    if param.grad is None:
            #        print(f"Parameter {name} has no gradient")
            #    else:
            #        print(f"Parameter {name} has gradient mean: {param.grad.mean().item()}")

            metrics['train_losses'].append(loss.item())
            print(f"[Step {train_step}] Train Loss: {loss.item():.4f}")
            train_step += 1 

        model.eval()
        with torch.no_grad():
            for batch in bunch.valid_dl:
                x, y = batch
                x, y = x.to(device), y.to(device)

                print(f"Input data range: {x.min().item()} to {x.max().item()}")
                print(f"Label data range: {y.min().item()} to {y.max().item()}")

                pred = model(x)
                val_loss = loss_fn(pred, y)

                metrics['valid_losses'].append(val_loss.item())
                print(f"[Step {valid_step}] Val Loss: {val_loss.item():.4f}")
                valid_step += 1

        visualise_predictions(model, bunch.valid_dl, device, args.output_dir, num_samples=5, num_classes=args.num_classes)

    return model, metrics

def trace_model(args, best_val_model):
    device = torch.device('cpu')
    if not best_val_model.endswith('.pt'):
        best_val_model += '.pt'

    print(f"Loading model from {best_val_model}...")
    model = load_model(best_val_model, args.num_classes, device)

    example_loader = SegmentationDataLoader(args.image_path, args.view, batch_size=args.batch_size, valid_pct=0.25, device=device)

    input_examples = None
    for batch in example_loader.train_dl:
        x, _ = batch
        input_examples = (x.cpu())
        break

    print("Tracing model...")
    traced_model = torch.jit.trace(model, input_examples)

    traced_model_save_path = f"{args.output_dir}/models/pass{args.vertex_pass}/{args.view}/traced_{args.model_name}.pt"
    traced_model.save(traced_model_save_path)

    print(f"Traced model saved to: {traced_model_save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="U-Net Vertex Finder Training")
    parser.add_argument("-i", "--image_path", type=str, required=True)
    parser.add_argument("-b", "--batch_size", type=int, default=64)
    parser.add_argument("-n", "--num_classes", type=int, default=4)
    parser.add_argument("-e", "--n_epochs", type=int, default=20)
    parser.add_argument("-m", "--model_name", type=str, required=True)
    parser.add_argument("-v", "--view", type=str, default="U")
    parser.add_argument("-p", "--vertex_pass", type=int, default=1)
    parser.add_argument("-s", "--seed", type=int, default=42)
    parser.add_argument("-o", "--output_dir", type=str, default="outputs")
    args = parser.parse_args()

    val_model, metrics = train_model(args)
    #trace_model(args, val_model)

    plot_loss(args.output_dir, metrics)