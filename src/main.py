import torch
import os
import argparse
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from dataset import SegmentationDataLoader
from utils import set_seed, create_model, get_class_weights, load_model, accuracy, save_model, plot_loss_accuracy, dice_coefficient, intersection_over_union
import numpy as np

def train_model(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    os.makedirs(f"{args.output_dir}/models/pass{args.vertex_pass}/{args.view}/", exist_ok=True)
    set_seed(args.seed)

    writer = SummaryWriter(log_dir=f"{args.output_dir}/logs/pass{args.vertex_pass}/{args.view}")

    bunch = SegmentationDataLoader(args.image_path, args.view, batch_size=args.batch_size, valid_pct=0.25, device=device)
    train_stats = bunch.count_classes(args.num_classes)
    weights = get_class_weights(train_stats)

    model, loss_fn, optim = create_model(args.num_classes, weights, device)
    model = model.to(device)

    metrics = {
        'train_losses': [], 'val_losses': [],
        'train_accs': [], 'val_accs': [],
        'train_dice_scores': [], 'val_dice_scores': [],
        'train_iou_scores': [], 'val_iou_scores': []
    }

    best_val_loss = float('inf')
    best_val_model = None

    set_seed(args.seed)
    step = 0 

    for e in tqdm(range(args.n_epochs), desc="Training"):
        model.train()

        for batch in bunch.train_dl:
            x, y = batch
            x, y = x.to(device), y.to(device)

            optim.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)

            loss.backward()
            optim.step()

            metrics['train_losses'].append(loss.item())
            metrics['train_accs'].append(accuracy(pred, y).cpu().numpy())
            metrics['train_dice_scores'].append(dice_coefficient(pred, y).cpu().numpy())
            metrics['train_iou_scores'].append(intersection_over_union(pred, y).cpu().numpy())

            writer.add_scalar('Loss/train', loss.item(), step)
            writer.add_scalar('Accuracy/train', accuracy(pred, y).cpu().numpy(), step)
            writer.add_scalar('Dice/train', dice_coefficient(pred, y).cpu().numpy(), step)
            writer.add_scalar('IoU/train', intersection_over_union(pred, y).cpu().numpy(), step)

            step += 1 

        model.eval()
        with torch.no_grad():
            for batch in bunch.valid_dl:
                x, y = batch
                x, y = x.to(device), y.to(device)

                pred = model(x)
                val_loss = loss_fn(pred, y)

                metrics['val_losses'].append(val_loss.item())
                metrics['val_accs'].append(accuracy(pred, y).cpu().numpy())
                metrics['val_dice_scores'].append(dice_coefficient(pred, y).cpu().numpy())
                metrics['val_iou_scores'].append(intersection_over_union(pred, y).cpu().numpy())

                writer.add_scalar('Loss/val', val_loss.item(), step)
                writer.add_scalar('Accuracy/val', accuracy(pred, y).cpu().numpy(), step)
                writer.add_scalar('Dice/val', dice_coefficient(pred, y).cpu().numpy(), step)
                writer.add_scalar('IoU/val', intersection_over_union(pred, y).cpu().numpy(), step)

        current_val_loss = np.mean(metrics['val_losses'][-len(bunch.valid_dl):])  
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            best_val_model = f"{args.output_dir}/models/pass{args.vertex_pass}/{args.view}/{args.model_name}_best"
            print(f"Saving model to: {best_val_model}.pt")
            save_model(model, x, best_val_model)

    writer.close()

    # Commented out the plot function
    # plot_loss_accuracy(
    #     metrics['train_losses'], metrics['val_losses'], 
    #     metrics['train_accs'], metrics['val_accs'], 
    #     metrics['train_dice_scores'], metrics['val_dice_scores'], 
    #     metrics['train_iou_scores'], metrics['val_iou_scores'], 
    #     step, args.output_dir
    # )

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

    val_model = train_model(args)
    #trace_model(args, val_model)
