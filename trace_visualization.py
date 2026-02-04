import argparse
import numpy as np
import matplotlib.pyplot as plt
from dataset import ASCADDataset

def get_args():
    parser = argparse.ArgumentParser(description="Visualize Raw SCA Trace")
    parser.add_argument("--data_path", type=str, required=True, help="Path to ASCAD.h5")
    parser.add_argument("--input_length", type=int, default=5000, help="Window size")
    parser.add_argument("--start_time", type=int, default=80000, help="Original start time index")
    parser.add_argument("--trace_idx", type=int, default=0, help="Index of the trace to plot (0 for first trace)")
    return parser.parse_args()

def normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def main():
    args = get_args()
    
    # 1. Load Data
    print(f"Loading trace {args.trace_idx} from {args.data_path}...")
    # We use 'test' split just to view the attack traces
    dataset = ASCADDataset(args.data_path, split='test', input_length=args.input_length)
    
    # Get the specific trace
    trace_tensor, _ = dataset[args.trace_idx]
    trace = trace_tensor.numpy()
    
    # Generate X-axis (Real Time)
    x_axis = np.arange(args.start_time, args.start_time + args.input_length)
    
    # 2. Plotting
    plt.figure(figsize=(15, 6))
    
    # Plot Raw Trace (Blue, higher alpha since it's the main focus now)
    # Using #1f77b4 which is the standard Matplotlib blue used in previous scripts
    plt.plot(x_axis, normalize(trace), label='Raw Power Trace', color='#1f77b4', linewidth=1.0, zorder=2)
    
    # 3. Highlight Standard Leakage Area (Yellow Box)
    leakage_start = 80945
    leakage_end = leakage_start + 1400  # 82345
    
    # Only draw if it's within our current view
    if (leakage_end > args.start_time) and (leakage_start < args.start_time + args.input_length):
        plt.axvspan(leakage_start, leakage_end, color='yellow', alpha=0.3, label='Standard Leakage Window', zorder=1)
        
        # Add Text Label
        mid_point = leakage_start + (leakage_end - leakage_start)/2
        if mid_point > args.start_time and mid_point < (args.start_time + args.input_length):
            plt.text(mid_point, 1.02, "Standard Leakage\n(S-Box Operation)", 
                     horizontalalignment='center', verticalalignment='bottom', 
                     fontsize=10, color='orange', fontweight='bold')

    # Styling
    plt.title(f"ASCAD Power Trace (Time: {args.start_time} - {args.start_time + args.input_length})")
    plt.xlabel("Original Time Sample")
    plt.ylabel("Normalized Amplitude")
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    out_file = "raw_trace_highlighted.png"
    plt.savefig(out_file)
    print(f"Plot saved to {out_file}")
    plt.show()

if __name__ == "__main__":
    main()