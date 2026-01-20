import argparse
import os
import math
import pickle
import sys
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Import your custom modules
from model import Transformer
import utils
from dataset import SCADataset

def get_args():
    parser = argparse.ArgumentParser(description="Side Channel Analysis Transformer")

    # Experiment / Data
    parser.add_argument("--data_path", type=str, required=True, help="Path to data file")
    parser.add_argument("--dataset", type=str, default="ASCAD", help="ASCAD or CHES20")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="Directory for checkpoints")
    parser.add_argument("--checkpoint_idx", type=int, default=0, help="Checkpoint index to restore")
    parser.add_argument("--result_path", type=str, default="results", help="Path for eval results")
    parser.add_argument("--do_train", action="store_true", help="Whether to perform training")
    parser.add_argument("--warm_start", type=bool, default=False, help="Warm start from checkpoint")
    
    # Hardware (Legacy flags maintained for script compatibility)
    parser.add_argument("--use_tpu", type=str, default="False", help="Ignored in PyTorch version")

    # Input Config
    parser.add_argument("--input_length", type=int, default=700, help="Input length for the model")
    parser.add_argument("--data_desync", type=int, default=0, help="Max trace desync for data augmentation")

    # Optimization
    parser.add_argument("--learning_rate", type=float, default=2.5e-4)
    parser.add_argument("--clip", type=float, default=0.25)
    parser.add_argument("--min_lr_ratio", type=float, default=0.004)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--train_steps", type=int, default=100000)
    parser.add_argument("--iterations", type=int, default=500, help="Log interval in steps")
    parser.add_argument("--save_steps", type=int, default=10000)
    parser.add_argument("--train_batch_size", type=int, default=256)
    parser.add_argument("--eval_batch_size", type=int, default=32)

    # Model Config
    parser.add_argument("--n_layer", type=int, default=6)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--d_head", type=int, default=32)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--d_inner", type=int, default=256)
    parser.add_argument("--n_head_softmax", type=int, default=4)
    parser.add_argument("--d_head_softmax", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--conv_kernel_size", type=int, default=3)
    parser.add_argument("--n_conv_layer", type=int, default=1)
    parser.add_argument("--pool_size", type=int, default=2)
    parser.add_argument("--d_kernel_map", type=int, default=128)
    parser.add_argument("--beta_hat_2", type=int, default=100)
    parser.add_argument("--model_normalization", type=str, default='preLC', choices=['preLC', 'postLC', 'none'])
    parser.add_argument("--head_initialization", type=str, default='forward', choices=['forward', 'backward', 'symmetric'])
    parser.add_argument("--softmax_attn", type=bool, default=True)
    
    # Evaluation Config
    parser.add_argument("--max_eval_batch", type=int, default=-1)
    parser.add_argument("--output_attn", type=bool, default=False)

    args = parser.parse_args()
    
    # Determine number of classes
    if args.dataset == 'CHES20':
        args.n_classes = 4 # Example specific to CHES20 logic
    else:
        args.n_classes = 256 # Standard for AES byte targets

    return args

# --- Custom LR Scheduler (Cosine Decay with Warmup) ---
def get_scheduler(optimizer, warmup_steps, train_steps, min_lr_ratio):
    def lr_lambda(current_step):
        current_step = float(current_step)
        if current_step < warmup_steps:
            return current_step / max(1.0, warmup_steps)
        
        # Cosine decay
        progress = float(current_step - warmup_steps) / float(max(1, train_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0) # Clip 0-1
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        
        return (1.0 - min_lr_ratio) * cosine_decay + min_lr_ratio

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# --- Infinite DataLoader for Step-based training ---
def cycle(iterable):
    while True:
        for x in iterable:
            yield x

def train(args, model, train_loader, test_loader, device):
    print(f"Starting training for {args.train_steps} steps...")
    
    # Optimizer & Scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = get_scheduler(optimizer, args.warmup_steps, args.train_steps, args.min_lr_ratio)
    
    if args.dataset == 'CHES20':
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    # Setup Checkpointing
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
        
    model.train()
    
    # Infinite iterator for step-based control
    train_iter = cycle(train_loader)
    
    running_loss = 0.0
    global_step = 0
    
    # Progress bar
    pbar = tqdm(total=args.train_steps)
    
    while global_step < args.train_steps:
        inputs, labels = next(train_iter)
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward
        output = model(inputs)
        # Handle tuple output if output_attn is enabled (though usually false in train)
        if isinstance(output, tuple):
             output = output[0]

        if args.dataset == 'CHES20':
            # CHES20 labels might need float conversion for BCE
            loss = criterion(output, labels.float())
        else:
            loss = criterion(output, labels)
            
        loss.backward()
        
        # Gradient Clipping
        nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        
        optimizer.step()
        scheduler.step()
        
        running_loss += loss.item()
        global_step += 1
        pbar.update(1)
        
        # Logging
        if global_step % args.iterations == 0:
            avg_loss = running_loss / args.iterations
            lr = scheduler.get_last_lr()[0]
            tqdm.write(f"Step [{global_step}/{args.train_steps}] | Loss: {avg_loss:.4f} | LR: {lr:.6f}")
            running_loss = 0.0
            
            # Simple Evaluation during training (Loss only)
            if test_loader is not None:
                model.eval()
                eval_loss = 0.0
                eval_batches = 0
                max_batches = args.max_eval_batch if args.max_eval_batch > 0 else len(test_loader)
                
                with torch.no_grad():
                    for i, (v_inp, v_lbl) in enumerate(test_loader):
                        if i >= max_batches: break
                        v_inp, v_lbl = v_inp.to(device), v_lbl.to(device)
                        v_out = model(v_inp)
                        if isinstance(v_out, tuple): v_out = v_out[0]
                        
                        if args.dataset == 'CHES20':
                            l = criterion(v_out, v_lbl.float())
                        else:
                            l = criterion(v_out, v_lbl)
                        eval_loss += l.item()
                        eval_batches += 1
                
                tqdm.write(f"Eval Loss: {eval_loss / eval_batches:.4f}")
                model.train()

        # Checkpointing
        if global_step % args.save_steps == 0:
            chk_name = os.path.join(args.checkpoint_dir, f"checkpoint_{global_step}.pt")
            torch.save({
                'step': global_step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict()
            }, chk_name)
            tqdm.write(f"Checkpoint saved: {chk_name}")

    # Final save
    final_path = os.path.join(args.checkpoint_dir, "model_final.pt")
    torch.save(model.state_dict(), final_path)
    print("Training finished.")


def evaluate_and_rank(args, model, test_loader, test_dataset, device):
    print("Starting evaluation...")
    model.eval()
    
    # Predict on Test Set
    all_preds = []
    max_batches = args.max_eval_batch if args.max_eval_batch > 0 else len(test_loader)
    
    with torch.no_grad():
        for i, (inputs, _) in enumerate(tqdm(test_loader, total=max_batches)):
            if i >= max_batches: break
            inputs = inputs.to(device)
            output = model(inputs)
            
            # If output_attn is True, model returns [scores, attn_list, softmax_score]
            if isinstance(output, list) or isinstance(output, tuple):
                scores = output[0]
                # If you need to save attention maps, do it here using output[1:]
            else:
                scores = output
                
            all_preds.append(scores.cpu().numpy())
            
    predictions = np.concatenate(all_preds, axis=0)
    
    # Load Metadata for Rank Calculation
    # We need to ensure we have exactly as many metadata entries as predictions
    n_preds = predictions.shape[0]
    
    if test_dataset.plaintexts is None or test_dataset.keys is None:
        print("Error: Metadata (plaintext/key) not found in dataset. Cannot compute rank.")
        return

    plaintexts = test_dataset.plaintexts[:n_preds]
    keys = test_dataset.keys[:n_preds]
    
    print(f"Computing key rank for {n_preds} traces...")
    
    # The original paper computes rank 100 times to average out shuffling noise
    key_rank_list = []
    for i in range(100):
        if args.dataset == 'ASCAD':
            # Utils must have compute_key_rank implemented
            ranks = utils.compute_key_rank(predictions, plaintexts, keys)
            key_rank_list.append(ranks)
        elif args.dataset == 'CHES20':
            # Implement CHES specific logic if needed
            pass
            
    key_ranks = np.stack(key_rank_list, axis=0)
    mean_ranks = np.mean(key_ranks, axis=0)
    
    # Save Results
    os.makedirs(os.path.dirname(args.result_path), exist_ok=True)
    txt_path = args.result_path + '.txt'
    
    with open(txt_path, 'w') as f:
        # Write individual run ranks
        for i in range(key_ranks.shape[0]):
            line = '\t'.join(map(str, key_ranks[i]))
            f.write(line + '\n')
        # Write mean rank
        mean_line = '\t'.join(map(str, mean_ranks))
        f.write(mean_line + '\n')
        
    print(f"Results saved to {txt_path}")
    print(f"Final Mean Rank (last trace): {mean_ranks[-1]}")


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Dataset Loading ---
    if args.dataset == 'ASCAD':
        if args.do_train:
            train_ds = SCADataset(
                data_path=args.data_path, 
                split='train', 
                input_length=args.input_length, 
                desync=args.data_desync
            )
            # Standard PyTorch Dataloader
            train_loader = DataLoader(
                train_ds, 
                batch_size=args.train_batch_size, 
                shuffle=True, 
                num_workers=4, 
                pin_memory=True
            )
        else:
            train_loader = None

        test_ds = SCADataset(
            data_path=args.data_path, 
            split='test', 
            input_length=args.input_length, 
            desync=0
        )
        test_loader = DataLoader(
            test_ds, 
            batch_size=args.eval_batch_size, 
            shuffle=False, 
            num_workers=4, 
            pin_memory=True
        )
    else:
        # Placeholder for CHES20 or other datasets
        raise NotImplementedError("Only ASCAD dataset is fully implemented in this script.")

    # --- Model Init ---
    model = Transformer(args).to(device)

    # --- Execution Mode ---
    if args.do_train:
        train(args, model, train_loader, test_loader, device)
    else:
        # Load Checkpoint for Evaluation
        if args.checkpoint_idx > 0:
            chk_path = os.path.join(args.checkpoint_dir, f"checkpoint_{args.checkpoint_idx}.pt")
        else:
            # Simple logic to find the 'best' or 'last' checkpoint would go here
            # For now, we assume a specific file or model_final.pt
            chk_path = os.path.join(args.checkpoint_dir, "model_final.pt")
            if not os.path.exists(chk_path):
                # Try finding any checkpoint
                chk_path = os.path.join(args.checkpoint_dir, f"checkpoint_{args.train_steps}.pt")

        if os.path.exists(chk_path):
            print(f"Loading checkpoint: {chk_path}")
            checkpoint = torch.load(chk_path, map_location=device)
            
            # Handle both full checkpoint dict and direct state_dict
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
        else:
            print(f"Warning: Checkpoint {chk_path} not found. Evaluating initialized model.")

        evaluate_and_rank(args, model, test_loader, test_ds, device)

if __name__ == "__main__":
    main()