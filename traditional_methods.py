import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from sklearn.feature_selection import mutual_info_classif

# Import your dataset
from dataset import ASCADDataset

# Lookup table for Hamming Weight (0-255)
HW = [bin(x).count('1') for x in range(256)]

def get_args():
    parser = argparse.ArgumentParser(description="Compare SCA Feature Selection Methods")
    parser.add_argument("--data_path", type=str, required=True, help="Path to ASCAD.h5")
    
    # Dataset Config
    parser.add_argument("--input_length", type=int, default=5000, help="Window size")
    parser.add_argument("--start_time", type=int, default=80000, help="Original start time index")
    parser.add_argument("--num_traces", type=int, default=2000, help="Number of traces to use")
    
    # Method Selection
    parser.add_argument("--method", type=str, default="all", 
                        choices=["snr", "dom", "mi", "pearson", "redundancy", "all"],
                        help="Which metric to run")
    
    # Accuracy Config
    parser.add_argument("--top_k", type=int, default=100, help="Number of top points to check for accuracy")
    
    return parser.parse_args()

def normalize(data):
    """Normalize data to 0-1 range for plotting."""
    if np.max(data) == np.min(data):
        return data
    return (data - np.min(data)) / (np.max(data) - np.min(data))

# --- METRICS IMPLEMENTATION ---

def compute_snr(traces, labels):
    print("Computing SNR...")
    classes = np.unique(labels)
    n_classes = len(classes)
    n_samples = traces.shape[1]
    
    means = np.zeros((n_classes, n_samples))
    variances = np.zeros((n_classes, n_samples))
    
    for i, c in enumerate(tqdm(classes, desc="SNR Classes")):
        class_traces = traces[labels == c]
        if len(class_traces) > 1:
            means[i] = np.mean(class_traces, axis=0)
            variances[i] = np.var(class_traces, axis=0)
            
    var_of_means = np.var(means, axis=0)
    mean_of_vars = np.mean(variances, axis=0)
    return var_of_means / (mean_of_vars + 1e-10)

def compute_dom(traces, labels):
    print("Computing DoM (Difference of Means)...")
    mask = 1 
    mean_0 = np.mean(traces[(labels & mask) == 0], axis=0)
    mean_1 = np.mean(traces[(labels & mask) == 1], axis=0)
    return np.abs(mean_0 - mean_1)

def compute_mi(traces, labels, chunk_size=100):
    print("Computing Mutual Information (MRMR Relevance)...")
    n_samples, n_features = traces.shape
    mi_scores = np.zeros(n_features)
    
    for i in tqdm(range(0, n_features, chunk_size), desc="Calculating MI"):
        end = min(i + chunk_size, n_features)
        chunk = traces[:, i:end]
        mi_scores[i:end] = mutual_info_classif(chunk, labels, discrete_features=False, random_state=42, n_neighbors=3)
        
    return mi_scores

def compute_pearson(traces, labels):
    print("Computing Pearson Correlation (CPA)...")
    hw_labels = np.array([HW[l] for l in labels])
    n_traces, n_samples = traces.shape
    
    y_centered = hw_labels - np.mean(hw_labels)
    std_y = np.std(hw_labels)
    
    mean_x = np.mean(traces, axis=0)
    std_x = np.std(traces, axis=0)
    
    x_centered = traces - mean_x
    covariance = np.dot(x_centered.T, y_centered) / n_traces
    
    correlations = covariance / (std_x * std_y + 1e-10)
    return np.abs(correlations)

def compute_redundancy(traces, labels):
    print("Computing Redundancy Map...")
    cpa = compute_pearson(traces, labels)
    best_idx = np.argmax(cpa)
    print(f"  Reference Point (Max CPA): Index {best_idx}")
    
    reference_col = traces[:, best_idx]
    n_traces, n_samples = traces.shape
    
    ref_centered = reference_col - np.mean(reference_col)
    std_ref = np.std(reference_col)
    
    mean_x = np.mean(traces, axis=0)
    std_x = np.std(traces, axis=0)
    x_centered = traces - mean_x
    
    covariance = np.dot(x_centered.T, ref_centered) / n_traces
    redundancy = covariance / (std_x * std_ref + 1e-10)
    return np.abs(redundancy)

# --- ACCURACY LOGIC ---

def calculate_accuracy(data, name, args):
    """Calculates if High-Value points fall within the Standard Leakage Window."""
    
    # Standard ASCAD Leakage: 80945 to 82345
    leakage_start_abs = 80945
    leakage_len = 1400
    
    rel_start = leakage_start_abs - args.start_time
    rel_end = rel_start + leakage_len
    
    # Check bounds
    if rel_start < 0 or rel_end > args.input_length:
        print(f"\n--- Accuracy Report ({name}) ---")
        print(f"Warning: Standard leakage window is outside current view. Accuracy 0%.")
        return

    print(f"\n--- Accuracy Report ({name}) ---")
    print(f"Highlighted Region: Abs {leakage_start_abs}-{leakage_start_abs+leakage_len}")
    
    # 1. Top-1 Analysis
    max_idx = np.argmax(data)
    is_inside_max = (rel_start <= max_idx < rel_end)
    
    print(f"[1] Top-1 Analysis (Max Peak)")
    print(f"    Location: {max_idx + args.start_time} (Index {max_idx})")
    print(f"    Inside Region? {'YES' if is_inside_max else 'NO'}")
    
    # 2. Top-K Accuracy
    top_k_indices = np.argsort(data)[-args.top_k:]
    count_inside = 0
    for idx in top_k_indices:
        if rel_start <= idx < rel_end:
            count_inside += 1
            
    accuracy = (count_inside / args.top_k) * 100.0
    
    print(f"[2] Top-{args.top_k} Accuracy")
    print(f"    Points in Region: {count_inside}/{args.top_k}")
    print(f"    Accuracy:         {accuracy:.2f}%")
    print("-" * 30)

# --- PLOTTING LOGIC ---

def plot_single(x_axis, metric_data, raw_trace, name, color, start, end, out_file):
    plt.figure(figsize=(15, 6))
    
    norm_metric = normalize(metric_data)
    norm_trace = normalize(raw_trace)
    
    plt.plot(x_axis, norm_metric, label=name, color=color, linewidth=1.5, zorder=3)
    plt.plot(x_axis, norm_trace, label='Raw Power Trace', alpha=0.3, color='#1f77b4', linestyle='-', zorder=2)
    
    leak_start, leak_end = 80945, 82345
    if (leak_end > start) and (leak_start < end):
        plt.axvspan(leak_start, leak_end, color='yellow', alpha=0.2, label='Std. Leakage', zorder=1)
        mid_point = leak_start + (leak_end - leak_start)/2
        if mid_point > start and mid_point < end:
            plt.text(mid_point, 1.02, "Standard Leakage", 
                     horizontalalignment='center', verticalalignment='bottom', 
                     fontsize=9, color='orange', fontweight='bold')

    plt.title(f"{name} vs Raw Trace ({start} - {end})")
    plt.xlabel("Original Time Sample")
    plt.ylabel("Normalized Score")
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_file)
    print(f"Saved {out_file}")
    plt.close()

def main():
    args = get_args()
    
    # 1. Load Data
    print(f"Loading {args.num_traces} traces from {args.data_path}...")
    dataset = ASCADDataset(args.data_path, split='test', input_length=args.input_length)
    
    traces_list, labels_list = [], []
    limit = min(len(dataset), args.num_traces)
    
    for i in range(limit):
        t, l = dataset[i]
        traces_list.append(t.numpy())
        labels_list.append(int(l))
        
    traces = np.stack(traces_list)
    labels = np.array(labels_list)
    
    example_trace = traces[0]
    x_axis = np.arange(args.start_time, args.start_time + args.input_length)
    end_time = args.start_time + args.input_length

    # 2. Run Methods & Calculate Accuracy
    results = {}
    
    # Helper to run and report
    def run_and_report(func, name, color):
        data = func(traces, labels)
        calculate_accuracy(data, name, args)
        results[name] = (data, color)

    if args.method in ['snr', 'all']:
        run_and_report(compute_snr, 'SNR', '#9467bd')
        
    if args.method in ['dom', 'all']:
        run_and_report(compute_dom, 'DoM (LSB)', '#17becf')
        
    if args.method in ['mi', 'all']:
        run_and_report(compute_mi, 'Mutual Info', '#2ca02c')
        
    if args.method in ['pearson', 'all']:
        run_and_report(compute_pearson, 'Pearson (CPA)', '#d62728')
        
    if args.method in ['redundancy', 'all']:
        run_and_report(compute_redundancy, 'Redundancy', '#ff7f0e')

    # 3. Plotting
    if args.method != 'all':
        name = list(results.keys())[0]
        data, _ = results[name]
        out_file = f"poi_result_{args.method}.png"
        plot_single(x_axis, data, example_trace, name, '#d62728', args.start_time, end_time, out_file)
    else:
        print("Generating Comparison Plot...")
        n_plots = len(results)
        plt.figure(figsize=(15, 4 * n_plots))
        norm_trace = normalize(example_trace)
        
        for i, (name, (data, color)) in enumerate(results.items()):
            plt.subplot(n_plots, 1, i+1)
            plt.plot(x_axis, normalize(data), color=color, label=name, linewidth=1.5, zorder=3)
            plt.plot(x_axis, norm_trace, label='Raw Power Trace', alpha=0.3, color='#1f77b4', linestyle='-', zorder=2)
            plt.axvspan(80945, 82345, color='yellow', alpha=0.2, label='Std. Leakage', zorder=1)
            plt.title(name)
            plt.grid(True, alpha=0.3)
            plt.legend(loc='upper right')
            
        plt.xlabel("Original Time Sample")
        plt.tight_layout()
        out_file = "poi_comparison_all.png"
        plt.savefig(out_file)
        print(f"Saved {out_file}")

if __name__ == "__main__":
    main()