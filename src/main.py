import torch
import os
import argparse
from tqdm import tqdm
from dataset import SegmentationDataLoader
from utils import set_seed, create_model, get_class_weights, load_model, accuracy, save_model, plot_loss_accuracy, plot_class_performance

def train_model(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    os.makedirs(f"{args.output_dir}/models/pass{args.vertex_pass}/{args.view}/", exist_ok=True)
    set_seed(args.seed)

    bunch = SegmentationDataLoader(args.image_path, args.view, batch_size=args.batch_size, valid_pct=0.25, device=device)
    train_stats = bunch.count_classes(args.num_classes)
    weights = get_class_weights(train_stats)

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    model, loss_fn, optim = create_model(args.num_classes, weights, device)
    model = model.to(device) 

    best_val_loss = float('inf')
    best_val_model = None
    class_names = ['Background'] + [f'Class_{i}' for i in range(1, args.num_classes)]

    for e in tqdm(range(args.n_epochs), desc="Training"):
        model.train()
        epoch_train_loss = 0.0
        epoch_train_acc = 0.0
        num_batches = 0

        for batch in bunch.train_dl:
            x, y = batch
            x, y = x.to(device), y.to(device)  
            pred = model(x)
            loss = loss_fn(pred, y)
            epoch_train_loss += loss.item()
            epoch_train_acc += accuracy(pred, y)
            loss.backward()
            optim.step()
            optim.zero_grad()
            num_batches += 1

        train_losses.append(epoch_train_loss / num_batches)
        train_accs.append(epoch_train_acc / num_batches)

        model.eval()
        epoch_val_loss = 0.0
        epoch_val_acc = 0.0
        num_val_batches = 0
        
        with torch.no_grad():
            for x, y in bunch.valid_dl:
                x, y = x.to(device), y.to(device)  
                pred = model(x)
                val_loss = loss_fn(pred, y)
                epoch_val_loss += val_loss.item()
                epoch_val_acc += accuracy(pred, y)
                num_val_batches += 1

        val_losses.append(epoch_val_loss / num_val_batches)
        val_accs.append(epoch_val_acc / num_val_batches)

        if (epoch_val_loss / num_val_batches) < best_val_loss:
            best_val_loss = epoch_val_loss / num_val_batches
            best_val_model = f"{args.output_dir}/models/pass{args.vertex_pass}/{args.view}/{args.model_name}_best"
            print(f"Saving model to: {best_val_model}.pt")  
            save_model(model, x, best_val_model)  

    plot_loss_accuracy(train_losses, val_losses, train_accs, val_accs, args.n_epochs, args.output_dir)
    
    return best_val_model


def trace_model(args, best_val_model):
    device = torch.device('cpu')  
    model = load_model(best_val_model, args.num_classes, device)
    
    example_loader = SegmentationDataLoader(args.image_path, args.view, batch_size=args.batch_size, valid_pct=0.25, device=device)
    
    for batch in example_loader.train_dl:
        example_input = batch[0].to(device)
        break
    
    traced_model = torch.jit.trace(model, example_input)
    
    traced_model_save_path = f"{args.output_dir}/models/pass{args.vertex_pass}/{args.view}/traced_{args.model_name}.pt"
    traced_model.save(traced_model_save_path)
    
    print(f"Traced model saved to: {traced_model_save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="U-Net Vertex Finder Training")
    parser.add_argument("-i", "--image_path", type=str, required=True)
    parser.add_argument("-b", "--batch_size", type=int, default=32)
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
