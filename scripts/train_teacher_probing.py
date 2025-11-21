#!/usr/bin/env python3
"""
train_teacher_probing.py

Stage 1: Teacher Semantic Extractor Preparation
Train the GNN semantic heads and action heads on a frozen teacher VLA model.

Usage:
    python scripts/train_teacher_probing.py --config configs/actdistill_stage1.yaml
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from vla import load_vla
from prismatic.vla import get_vla_dataset_and_collator


def move_pixel_values_to_device(pixel_values, device):
    if isinstance(pixel_values, torch.Tensor):
        return pixel_values.to(device)
    elif isinstance(pixel_values, dict):
        return {k: move_pixel_values_to_device(v, device) for k, v in pixel_values.items()}
    elif pixel_values is None:
        return None
    else:
        raise ValueError(f"Unsupported pixel_values type: {type(pixel_values)}")


def setup_environment():
    """Set up environment variables for Stage 1 training."""
    os.environ['USE_ACTDISTILL'] = 'TRUE'
    os.environ['SEMANTIC_DIM'] = '512'
    os.environ['GNN_TYPE'] = 'GAT'


def freeze_backbone(model):
    """
    Freeze the entire model except GNN semantic heads and action heads.
    """
    print("[Stage 1] Freezing backbone parameters...")

    # Freeze all parameters first
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze GNN semantic heads and action heads in LLM
    if hasattr(model, 'vlm') and hasattr(model.vlm, 'llm_backbone'):
        llm = model.vlm.llm_backbone.llm

        if hasattr(llm, 'gnn_semantic_heads') and llm.gnn_semantic_heads is not None:
            print("[Stage 1] Unfreezing GNN semantic heads...")
            for param in llm.gnn_semantic_heads.parameters():
                param.requires_grad = True

        if hasattr(llm, 'action_heads') and llm.action_heads is not None:
            print("[Stage 1] Unfreezing action prediction heads...")
            for param in llm.action_heads.parameters():
                param.requires_grad = True

    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[Stage 1] Trainable parameters: {trainable_params:,} / {total_params:,} "
          f"({100 * trainable_params / total_params:.2f}%)")


def compute_preparation_loss(semantics, actions, action_gt):
    """
    Compute preparation loss: L_prep = Σ_l MSE(action_pred_l, action_gt)

    This loss trains the GNN heads to extract semantics that can predict actions.

    Args:
        semantics: List of [batch, semantic_dim] tensors from each layer
        actions: List of [batch, 7] action predictions from each layer
        action_gt: [batch, 7] ground truth action

    Returns:
        Scalar loss
    """
    total_loss = 0.0
    num_layers = len(actions)

    for l in range(num_layers):
        # MSE between predicted action and ground truth
        layer_loss = F.mse_loss(actions[l], action_gt)
        total_loss += layer_loss

    # Average over layers
    return total_loss / num_layers


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    device,
    epoch: int,
    total_epochs: int
):
    """Train for one epoch."""
    model.train()

    total_loss = 0.0
    num_batches = 0

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{total_epochs}")

    for batch_idx, batch in enumerate(progress_bar):
        # Move batch to device
        pixel_values = move_pixel_values_to_device(batch['pixel_values'], device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        actions = batch['actions'].to(device)

        # Extract ground truth action (last timestep)
        action_gt = actions[:, -1, :]  # [batch, 7]

        # Forward pass through VLM (teacher model)
        optimizer.zero_grad()

        try:
            vlm_output = model.vlm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                labels=input_ids,  # Not used but required for interface
                output_hidden_states=True,
                return_dict=True,
            )

            # Unpack outputs
            if isinstance(vlm_output, tuple) and len(vlm_output) == 3:
                output, semantics, action_preds = vlm_output
            else:
                print(f"[Warning] Unexpected output format from VLM. Skipping batch.")
                continue

            # Compute preparation loss
            loss = compute_preparation_loss(semantics, action_preds, action_gt)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Update statistics
            total_loss += loss.item()
            num_batches += 1

            # Update progress bar
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        except Exception as e:
            print(f"[Error] Exception during training: {e}")
            continue

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    print(f"[Stage 1] Epoch {epoch + 1} - Average Loss: {avg_loss:.4f}")

    return avg_loss


def save_semantic_heads(model, save_path: Path):
    """
    Save only the GNN semantic heads and action heads.
    """
    save_dict = {}

    if hasattr(model, 'vlm') and hasattr(model.vlm, 'llm_backbone'):
        llm = model.vlm.llm_backbone.llm

        if hasattr(llm, 'gnn_semantic_heads') and llm.gnn_semantic_heads is not None:
            save_dict['gnn_semantic_heads'] = llm.gnn_semantic_heads.state_dict()

        if hasattr(llm, 'action_heads') and llm.action_heads is not None:
            save_dict['action_heads'] = llm.action_heads.state_dict()

    torch.save(save_dict, save_path)
    print(f"[Stage 1] Saved semantic heads to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="ActDistill Stage 1: Teacher Semantic Probing")
    parser.add_argument('--teacher_checkpoint', type=str, required=True,
                        help='Path to pretrained teacher VLA checkpoint')
    parser.add_argument('--data_root_dir', type=str, required=True,
                        help='Root directory of training data')
    parser.add_argument('--data_mix', type=str, default='oxe_magic_soup_plus_minus',
                        help='Open-X Embodiment mixture identifier')
    parser.add_argument('--output_dir', type=str, default='./checkpoints/actdistill_stage1',
                        help='Directory to save semantic heads')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--future_action_window_size', type=int, default=15,
                        help='Future action horizon used by dataset loader')
    parser.add_argument('--past_action_window_size', type=int, default=0,
                        help='Past action window size used by dataset loader')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of dataloader workers')
    parser.add_argument('--image_aug', action='store_true',
                        help='Enable image augmentation when loading teacher data')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to train on')

    args = parser.parse_args()

    # Setup
    setup_environment()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("ActDistill Stage 1: Teacher Semantic Extractor Preparation")
    print("=" * 80)
    print(f"Teacher checkpoint: {args.teacher_checkpoint}")
    print(f"Data directory: {args.data_root_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Device: {device}")
    print("=" * 80)

    # Load teacher model
    print("\n[Stage 1] Loading teacher VLA model...")
    model = load_vla(
        args.teacher_checkpoint,
        load_for_training=True,
        action_model_type='DiT-B',
        future_action_window_size=15,
    )
    model = model.to(device)

    # Freeze backbone, unfreeze semantic heads
    freeze_backbone(model)

    # Setup optimizer (only for trainable parameters)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)

    print("\n[Stage 1] Loading dataset...")
    dataset, _, collator = get_vla_dataset_and_collator(
        args.data_root_dir,
        args.data_mix,
        image_transform=model.vlm.vision_backbone.get_image_transform(),
        tokenizer=model.vlm.llm_backbone.get_tokenizer(),
        prompt_builder_fn=model.vlm.llm_backbone.prompt_builder_fn,
        default_image_resolution=model.vlm.vision_backbone.default_image_resolution,
        image_aug=args.image_aug,
        load_all_data_for_training=True,
        future_action_window_size=args.future_action_window_size,
        past_action_window_size=args.past_action_window_size,
    )

    # CRITICAL: Use shuffle=False in BOTH training and caching to ensure consistency
    # Stage 2 will rely on the same data order to load cached teacher outputs
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False, 
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collator,
    )

    cache_dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,  # Same as training dataloader
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collator,
    )

    # Training loop
    print("\n[Stage 1] Starting training...")
    for epoch in range(args.epochs):
        avg_loss = train_one_epoch(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            total_epochs=args.epochs
        )

        # Save checkpoint every epoch
        checkpoint_path = output_dir / f"semantic_heads_epoch{epoch + 1}.pth"
        save_semantic_heads(model, checkpoint_path)

    # Save final checkpoint
    final_path = output_dir / "semantic_heads_final.pth"
    save_semantic_heads(model, final_path)
    print(f"\n[Stage 1] Training completed! Final checkpoint saved to {final_path}")

    # === CRITICAL: Extract and cache teacher outputs for Stage 2 ===
    print("\n" + "=" * 80)
    print("[Stage 1] Extracting teacher outputs for entire dataset...")
    print("=" * 80)
    
    cache_dir = output_dir / "teacher_outputs_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    extract_teacher_outputs(
        model=model,
        dataloader=cache_dataloader,
        cache_dir=cache_dir,
        device=device
    )
    
    print(f"\n[Stage 1] Teacher outputs cached to {cache_dir}")
    print("[Stage 1] Stage 1 completed! You can now run Stage 2 with cached teacher outputs.")


def extract_teacher_outputs(model, dataloader, cache_dir: Path, device):
    """
    Extract teacher outputs (semantics + actions) for the entire dataset
    and save them to disk for Stage 2 distillation.

    This eliminates the need to run teacher forward during Stage 2,
    significantly reducing memory and computation costs.
    """
    # Use train mode to enable GNN heads, but no_grad to skip gradient computation
    model.train()

    # Store all outputs
    all_semantics = []  # List of (batch_idx, layer_idx, semantics)
    all_actions = []    # List of (batch_idx, layer_idx, actions)

    print(f"[Stage 1] Processing {len(dataloader)} batches...")
    num_layers = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Extracting teacher outputs")):
            pixel_values = move_pixel_values_to_device(batch['pixel_values'], device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            try:
                # Forward pass through teacher
                vlm_output = model.vlm(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    labels=input_ids,
                    output_hidden_states=True,
                    return_dict=True,
                )
                
                # Unpack outputs
                if isinstance(vlm_output, tuple) and len(vlm_output) == 3:
                    output, semantics, action_preds = vlm_output
                    
                    if num_layers == 0:
                        num_layers = len(semantics)

                    # Save semantics for each layer
                    for layer_idx, sem in enumerate(semantics):
                        # sem: [batch, semantic_dim]
                        cache_file = cache_dir / f"batch{batch_idx:06d}_layer{layer_idx:02d}_semantic.pt"
                        torch.save(sem.cpu(), cache_file)
                    
                    # Save action predictions for each layer
                    for layer_idx, act in enumerate(action_preds):
                        # act: [batch, 7]
                        cache_file = cache_dir / f"batch{batch_idx:06d}_layer{layer_idx:02d}_action.pt"
                        torch.save(act.cpu(), cache_file)
                    
                    if batch_idx % 100 == 0:
                        print(f"[Stage 1] Cached {batch_idx}/{len(dataloader)} batches")
                        
                else:
                    print(f"[Warning] Unexpected output format for batch {batch_idx}")
                    continue
                    
            except Exception as e:
                print(f"[Error] Failed to process batch {batch_idx}: {e}")
                continue
    
    # Save metadata
    metadata = {
        'num_batches': len(dataloader),
        'num_layers': num_layers,
        'batch_size': dataloader.batch_size,
    }
    torch.save(metadata, cache_dir / "metadata.pt")
    
    print(f"[Stage 1] Successfully cached {len(dataloader)} batches")
    print(f"[Stage 1] Metadata: {metadata}")


if __name__ == '__main__':
    main()
