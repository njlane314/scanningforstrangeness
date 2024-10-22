import torch
import os
import argparse
from tqdm import tqdm
from dataset import SegmentationDataLoader
from utils import set_seed, create_model, get_class_weights, load_model, accuracy, save_model, plot_loss_accuracy, plot_class_performance

def train_model(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    os.makedirs(f"{args.output_dir}/models/pass{args.vertex_pass}/{args.view}/", exist_ok=True)
    set_seed(args.seed)

    bunch = SegmentationDataLoader(args.image_path, args.view, batch_size=args.batch_size, valid_pct=0.25, device=device)
    train_stats = bunch.count_classes(args.num_classes)
    weights = get_class_weights(train_stats)

    train_losses = torch.zeros(args.n_epochs * len(bunch.train_dl), device=device)
    val_losses = torch.zeros(args.n_epochs, device=device)
    train_accs = torch.zeros(args.n_epochs * len(bunch.train_dl), device=device)
    val_accs = torch.zeros(args.n_epochs, device=device)

    model, loss_fn, optim = create_model(args.num_classes, weights, device)
    model = model.to(device) 

    best_val_loss = float('inf')
    best_val_model = None
    class_names = ['Background'] + [f'Class_{i}' for i in range(1, args.num_classes)]

    i = 0
    for e in tqdm(range(args.n_epochs), desc="Training"):
        model.train()
        for batch in bunch.train_dl:
            x, y = batch
            x, y = x.to(device), y.to(device)  
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
                x, y = x.to(device), y.to(device)  
                pred = model(x)  
                val_loss.append(loss_fn(pred, y).item())
                val_acc.append(accuracy(pred, y))
            avg_val_loss = torch.mean(torch.tensor(val_loss, device=device))
            val_losses[e] = avg_val_loss
            val_accs[e] = torch.mean(torch.tensor(val_acc, device=device))

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_val_model = f"{args.output_dir}/models/pass{args.vertex_pass}/{args.view}/{args.model_name}_best.pt"
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
