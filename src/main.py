import torch
import os
import argparse
from tqdm import tqdm
from dataset import SegmentationDataLoader
from utils import set_seed, create_model, get_class_weights, load_model, accuracy, save_model, plot_loss_accuracy, dice_coefficient, intersection_over_union
import numpy as np

def train_model(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    os.makedirs(f"{args.output_dir}/models/pass{args.vertex_pass}/{args.view}/", exist_ok=True)
    set_seed(args.seed)

    bunch = SegmentationDataLoader(args.image_path, args.view, batch_size=args.batch_size, valid_pct=0.25, device=device)
    train_stats = bunch.count_classes(args.num_classes)
    weights = get_class_weights(train_stats)

    model, loss_fn, optim = create_model(args.num_classes, weights, device)
    model = model.to(device)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min', factor=0.1, patience=3)

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    train_dice_scores, val_dice_scores = []
    train_iou_scores, val_iou_scores = []

    batch_num = 0
    best_val_loss = float('inf')
    best_val_model = None

    for e in tqdm(range(args.n_epochs), desc="Training"):
        model.train()
        epoch_train_loss, epoch_train_acc = 0.0, 0.0
        epoch_train_dice, epoch_train_iou = 0.0, 0.0

        for batch in bunch.train_dl:
            x, y = batch
            x, y = x.to(device), y.to(device)

            optim.zero_grad()

            pred = model(x)
            loss = loss_fn(pred, y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()

            epoch_train_loss = loss.item()
            epoch_train_acc = accuracy(pred, y).cpu().numpy()
            epoch_train_dice = dice_coefficient(pred, y).cpu().numpy()
            epoch_train_iou = intersection_over_union(pred, y).cpu().numpy()

            train_losses.append(epoch_train_loss)
            train_accs.append(epoch_train_acc)
            train_dice_scores.append(epoch_train_dice)
            train_iou_scores.append(epoch_train_iou)

            batch_num += 1

        model.eval()
        epoch_val_loss, epoch_val_acc = 0.0, 0.0
        epoch_val_dice, epoch_val_iou = 0.0, 0.0
        num_val_batches = 0

        with torch.no_grad():
            for x, y in bunch.valid_dl:
                x, y = x.to(device), y.to(device)

                pred = model(x)
                val_loss = loss_fn(pred, y)

                epoch_val_loss = val_loss.item()
                epoch_val_acc = accuracy(pred, y).cpu().numpy()
                epoch_val_dice = dice_coefficient(pred, y).cpu().numpy()
                epoch_val_iou = intersection_over_union(pred, y).cpu().numpy()

                val_losses.append(epoch_val_loss)
                val_accs.append(epoch_val_acc)
                val_dice_scores.append(epoch_val_dice)
                val_iou_scores.append(epoch_val_iou)

                num_val_batches += 1

        if (epoch_val_loss / num_val_batches) < best_val_loss:
            best_val_loss = epoch_val_loss / num_val_batches
            best_val_model = f"{args.output_dir}/models/pass{args.vertex_pass}/{args.view}/{args.model_name}_best"
            print(f"Saving model to: {best_val_model}.pt")
            save_model(model, x, best_val_model)

        print(f"Epoch {e+1}/{args.n_epochs} - Train Dice: {train_dice_scores[-1]:.4f} - Val Dice: {val_dice_scores[-1]:.4f}")
        print(f"Epoch {e+1}/{args.n_epochs} - Train IoU: {train_iou_scores[-1]:.4f} - Val IoU: {val_iou_scores[-1]:.4f}")

        scheduler.step(np.mean(val_losses))

    plot_loss_accuracy(train_losses, val_losses, train_accs, val_accs, train_dice_scores, val_dice_scores, train_iou_scores, val_iou_scores, batch_num, args.output_dir)

    return best_val_model


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
    trace_model(args, val_model)
