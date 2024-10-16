from model import UNet
from data_loader import SegmentationDataLoader
import numpy as np
import torch
import argparse

def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)

def load_model(filename, num_classes, device):
    model = UNet(1, n_classes=num_classes, depth=4, n_filters=16, y_range=(0, num_classes - 1))
    model.load_state_dict(torch.load(filename, map_location=device))
    model.to(device)
    model.eval()
    return model

def process_view(input_dir, vertex_pass, view, num_classes, batch_size=32, seed=42):
    set_seed(seed)
    
    image_path = f"{input_dir}/Images_{view}/"
    device = torch.device('cpu')
    
    model_filename = f"{input_dir}/outputs/models/pass{vertex_pass}/{view}/uboone_hd_accel_19.pt"
    output_filename = f"pytorch_1_0_1_uboone_hd_accel_19.pt"

    bunch = SegmentationDataLoader(image_path, "Hits", "Truth", batch_size=batch_size, valid_pct=0.15, device=device)
    
    model = load_model(model_filename, num_classes, device)

    for batch in bunch.train_dl:
        input_examples = batch[0].to(device)
        break

    traced_model = torch.jit.trace(model, input_examples)
    traced_model.save(output_filename)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--vertex_pass", type=int, required=True)
    parser.add_argument("--n_classes", type=int, required=True)
    args = parser.parse_args()

    for view in ["U", "V", "W"]:
        process_view(args.input_dir, args.vertex_pass, view, args.n_classes)
