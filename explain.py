import argparse
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os

# Import your existing modules
from model import Transformer
from dataset import ASCADDataset

# Try importing Captum
try:
    from captum.attr import IntegratedGradients, Saliency, NoiseTunnel, LRP
    CAPTUM_AVAILABLE = True
except ImportError:
    CAPTUM_AVAILABLE = False
    print("Warning: Captum not installed. Only manual Saliency Map will be available.")

def get_args():
    parser = argparse.ArgumentParser(description="XAI for SCA Transformer")
    parser.add_argument("--data_path", type=str, required=True, help="Path to ASCAD.h5")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained .pt model")
    
    # Dataset Config
    parser.add_argument("--input_length", type=int, default=5000, help="Window size (e.g. 5000)")
    parser.add_argument("--start_time", type=int, default=80000, help="Original start time index (e.g. 80000)")
    
    # Model Architecture Args (MUST match the trained model)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_layer", type=int, default=6)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--d_head", type=int, default=32)
    parser.add_argument("--d_inner", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--n_classes", type=int, default=256)
    
    # Architecture specifics
    parser.add_argument("--n_conv_layer", type=int, default=1)
    parser.add_argument("--conv_kernel_size", type=int, default=3)
    parser.add_argument("--pool_size", type=int, default=2)
    parser.add_argument("--d_kernel_map", type=int, default=128)
    parser.add_argument("--beta_hat_2", type=int, default=100)
    parser.add_argument("--model_normalization", type=str, default='preLC')
    parser.add_argument("--head_initialization", type=str, default='forward')
    parser.add_argument("--softmax_attn", type=bool, default=True)
    parser.add_argument("--n_head_softmax", type=int, default=4)
    parser.add_argument("--d_head_softmax", type=int, default=32)
    parser.add_argument("--output_attn", type=bool, default=False) 
    
    # XAI Config
    parser.add_argument("--method", type=str, default="ig", 
                        choices=["ig", "saliency", "smoothgrad", "lrp"], 
                        help="Choose XAI Method")
    parser.add_argument("--num_traces", type=int, default=100, help="Number of traces to average")
    parser.add_argument("--batch_size", type=int, default=10, help="Internal batch size to prevent OOM")
    parser.add_argument("--top_k", type=int, default=100, help="Number of top points to check for accuracy")
    
    return parser.parse_args()

def load_model(args, device):
    model = Transformer(args).to(device)
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model not found at {args.model_path}")
    
    print(f"Loading model from {args.model_path}...")
    checkpoint = torch.load(args.model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    return model

def compute_manual_saliency(model, input_tensor, target_label):
    model.zero_grad()
    input_tensor.requires_grad = True
    output = model(input_tensor)
    target_scores = output.gather(1, target_label.view(-1, 1)).squeeze()
    target_scores.sum().backward()
    return input_tensor.grad.abs().detach().cpu().numpy()

# --- HELPER FUNCTION FOR MANUAL BATCHING ---
def batched_attribute(attr_method, input_tensor, target_label, batch_size, **kwargs):
    """
    Manually splits data into chunks and processes them one by one.
    This fixes OOM errors for methods like Saliency/LRP that lack internal_batch_size.
    """
    n = input_tensor.shape[0]
    result_list = []
    
    # Loop over the data in small chunks
    for i in range(0, n, batch_size):
        end = min(i + batch_size, n)
        batch_input = input_tensor[i:end]
        batch_target = target_label[i:end]
        
        # Run the attribute method on just this small chunk
        attr = attr_method.attribute(batch_input, target=batch_target, **kwargs)
        result_list.append(attr.detach().cpu())
        
    return torch.cat(result_list, dim=0).numpy()

def calculate_accuracy(avg_attribution, args):
    leakage_start_abs = 80945
    leakage_len = 1400
    
    rel_start = leakage_start_abs - args.start_time
    rel_end = rel_start + leakage_len
    
    if rel_start < 0 or rel_end > args.input_length:
        print(f"\nWarning: Standard leakage window is outside current view. Accuracy is 0%.")
        return

    print(f"\n--- Accuracy Report ({args.method.upper()}) ---")
    print(f"Highlighted Region: Abs {leakage_start_abs}-{leakage_start_abs+leakage_len}")
    
    max_idx = np.argmax(avg_attribution)
    is_inside_max = (rel_start <= max_idx < rel_end)
    
    print(f"\n[1] Top-1 Analysis (Most Important POI)")
    print(f"    Max Peak Location: {max_idx + args.start_time} (Index {max_idx})")
    print(f"    Inside Region?     {'YES' if is_inside_max else 'NO'}")
    
    top_k_indices = np.argsort(avg_attribution)[-args.top_k:]
    count_inside = 0
    for idx in top_k_indices:
        if rel_start <= idx < rel_end:
            count_inside += 1
            
    accuracy = (count_inside / args.top_k) * 100.0
    print(f"\n[2] Top-{args.top_k} Accuracy")
    print(f"    Points in Region: {count_inside}/{args.top_k}")
    print(f"    Accuracy:         {accuracy:.2f}%")

def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Data
    dataset = ASCADDataset(args.data_path, split='test', input_length=args.input_length)
    
    # 2. Load Model
    model = load_model(args, device)
    
    # 3. Select Data
    indices = range(args.num_traces)
    traces, labels = [], []
    for i in indices:
        t, l = dataset[i]
        traces.append(t)
        labels.append(l)
        
    input_tensor = torch.stack(traces).to(device)
    target_label = torch.stack(labels).to(device)
    
    print(f"Explaining {args.num_traces} traces using {args.method.upper()}...")

    # 4. Run XAI Method
    attributions = None
    
    if CAPTUM_AVAILABLE:
        if args.method == 'ig':
            # Integrated Gradients supports internal_batch_size NATIVELY
            ig = IntegratedGradients(model)
            attributions, _ = ig.attribute(input_tensor, target=target_label, n_steps=50, 
                                           return_convergence_delta=True, 
                                           internal_batch_size=args.batch_size)
            attributions = attributions.detach().cpu().numpy()
            
        elif args.method == 'saliency':
            # Saliency DOES NOT -> Use Helper
            sal = Saliency(model)
            attributions = batched_attribute(sal, input_tensor, target_label, args.batch_size, abs=True)
            
        elif args.method == 'smoothgrad':
            # SmoothGrad (NoiseTunnel) supports internal_batch_size NATIVELY
            print("  -> Applying SmoothGrad (50 noise samples)...")
            sal = Saliency(model)
            nt = NoiseTunnel(sal)
            attributions = nt.attribute(input_tensor, target=target_label, 
                                        nt_type='smoothgrad', nt_samples=50, stdevs=0.15, abs=True)
                                        #internal_batch_size=args.batch_size)
            attributions = attributions.detach().cpu().numpy()
            
        elif args.method == 'lrp':
            # LRP DOES NOT -> Use Helper
            print("  -> Applying LRP...")
            try:
                lrp = LRP(model)
                attributions = batched_attribute(lrp, input_tensor, target_label, args.batch_size)
            except Exception as e:
                print(f"\n[ERROR] LRP Failed: {e}")
                print("Falling back to Saliency Map...")
                sal = Saliency(model)
                attributions = batched_attribute(sal, input_tensor, target_label, args.batch_size, abs=True)

    if attributions is None:
        attributions = compute_manual_saliency(model, input_tensor, target_label)

    # 5. Process & Plot
    avg_attribution = np.mean(attributions, axis=0)
    avg_attribution = (avg_attribution - avg_attribution.min()) / (avg_attribution.max() - avg_attribution.min())
    
    calculate_accuracy(avg_attribution, args)

    x_axis = np.arange(args.start_time, args.start_time + args.input_length)
    example_trace = traces[0].numpy()
    norm_trace = (example_trace - example_trace.min()) / (example_trace.max() - example_trace.min())

    plt.figure(figsize=(15, 6))
    plt.plot(x_axis, avg_attribution, label=f'{args.method.upper()} Importance', color='#d62728', linewidth=1.5, zorder=3)
    plt.plot(x_axis, norm_trace, label='Raw Power Trace', alpha=0.3, color='#1f77b4', linestyle='-', zorder=2)
    
    leakage_start, leakage_end = 80945, 82345
    if (leakage_end > args.start_time) and (leakage_start < args.start_time + args.input_length):
        plt.axvspan(leakage_start, leakage_end, color='yellow', alpha=0.2, label='Standard Leakage', zorder=1)
        mid_point = leakage_start + (leakage_end - leakage_start)/2
        if mid_point > args.start_time and mid_point < (args.start_time + args.input_length):
            plt.text(mid_point, 1.02, "Standard Leakage", 
                     horizontalalignment='center', verticalalignment='bottom', 
                     fontsize=9, color='orange', fontweight='bold')

    plt.title(f"SCA Attribution Map: {args.method.upper()}")
    plt.xlabel("Original Time Sample")
    plt.ylabel("Normalized Importance")
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_file = f"poi_highlighted_{args.method}.png"
    plt.savefig(out_file)
    print(f"Result saved to {out_file}")
    plt.show()

if __name__ == "__main__":
    main()