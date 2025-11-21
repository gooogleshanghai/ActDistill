"""
cached_teacher_dataset.py

Dataset wrapper that loads pre-computed teacher outputs from disk.
Used in Stage 2 to avoid running teacher forward pass during training.
"""

import torch
from pathlib import Path
from typing import Dict, Any, Optional
from torch.utils.data import Dataset


class CachedTeacherDataset(Dataset):
    """
    Wraps an existing dataset and augments each sample with pre-computed
    teacher outputs (semantics + actions) loaded from disk.
    
    This enables offline distillation where teacher outputs are computed
    once in Stage 1 and reused throughout Stage 2 training.
    """
    
    def __init__(
        self,
        base_dataset: Dataset,
        teacher_cache_dir: str,
        num_layers: int = 32,
    ):
       
        self.base_dataset = base_dataset
        self.cache_dir = Path(teacher_cache_dir)
        self.num_layers = num_layers
        self.dataset_statistics = getattr(self.base_dataset, 'dataset_statistics', None)
        
        # Validate cache directory exists
        if not self.cache_dir.exists():
            raise ValueError(f"Teacher cache directory not found: {self.cache_dir}")
        
        # Load and validate metadata
        metadata_path = self.cache_dir / "metadata.pt"
        if metadata_path.exists():
            self.metadata = torch.load(metadata_path)
            print(f"[CachedTeacherDataset] Loaded metadata: {self.metadata}")
        else:
            print(f"[Warning] No metadata found at {metadata_path}")
            self.metadata = {}
        
        # Validate cache consistency
        self._validate_cache_consistency()
            
        print(f"[CachedTeacherDataset] Initialized with cache: {self.cache_dir}")
        print(f"[CachedTeacherDataset] Base dataset size: {len(self.base_dataset)}")
        print(f"[CachedTeacherDataset] Expected cache batches: {self.metadata.get('num_batches', 'unknown')}")
    
    def _validate_cache_consistency(self):
        """Validate that cache files are consistent with dataset size and metadata."""
        if not self.metadata:
            print("[Warning] No metadata available, skipping validation")
            return
        
        expected_batches = self.metadata.get('num_batches')
        expected_batch_size = self.metadata.get('batch_size')
        expected_layers = self.metadata.get('num_layers')
        
        if expected_batches is None:
            print("[Warning] No batch count in metadata")
            return
            
        if expected_batch_size is None:
            print("[Warning] No batch size in metadata")
            return
            
        if expected_layers is None:
            print("[Warning] No layer count in metadata")
            return
        
        # Check data alignment
        expected_dataset_size = expected_batches * expected_batch_size
        actual_dataset_size = len(self.base_dataset)
        
        if actual_dataset_size != expected_dataset_size:
            raise ValueError(
                f"Dataset size mismatch! Base dataset has {actual_dataset_size} samples, "
                f"but cache expects {expected_dataset_size} samples ({expected_batches} batches × {expected_batch_size} batch_size). "
                f"This indicates Stage 1 and Stage 2 use different data orderings."
            )
        
        # Validate cache files exist
        missing_files = []
        for batch_idx in range(expected_batches):
            for layer_idx in range(expected_layers):
                sem_file = self.cache_dir / f"batch{batch_idx:06d}_layer{layer_idx:02d}_semantic.pt"
                act_file = self.cache_dir / f"batch{batch_idx:06d}_layer{layer_idx:02d}_action.pt"
                
                if not sem_file.exists():
                    missing_files.append(str(sem_file))
                if not act_file.exists():
                    missing_files.append(str(act_file))
        
        if missing_files:
            raise FileNotFoundError(
                f"Missing {len(missing_files)} cache files. First few missing files:\n" + 
                "\n".join(missing_files[:5]) +
                ("\n..." if len(missing_files) > 5 else "")
            )
        
        print(f"[CachedTeacherDataset] Cache validation passed: {expected_batches} batches × {expected_layers} layers × {expected_batch_size} batch_size")
        
        # Check layer consistency
        if expected_layers != self.num_layers:
            print(f"[Warning] Layer count mismatch: metadata says {expected_layers}, but model expects {self.num_layers}")
    
    def __len__(self):
        return len(self.base_dataset)

    def __getattr__(self, name):
        """Delegate attribute lookups to the wrapped dataset for sampler utilities."""
        if name in {"base_dataset", "cache_dir", "num_layers", "dataset_statistics", "metadata"}:
            raise AttributeError
        return getattr(self.base_dataset, name)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Returns:
            dict with keys:
                - All original dataset keys (pixel_values, input_ids, etc.)
                - teacher_semantics: List[Tensor] of [batch, semantic_dim]
                - teacher_actions: List[Tensor] of [batch, 7]
        """
        # Get original sample
        sample = self.base_dataset[idx]
        
        # Determine batch index for this sample
        # Assumes sequential batching in Stage 1
        batch_size = self.metadata.get('batch_size', 64)
        batch_idx = idx // batch_size
        
        # Load teacher outputs for this batch
        teacher_semantics = []
        teacher_actions = []
        
        for layer_idx in range(self.num_layers):
            # Load semantic features
            sem_file = self.cache_dir / f"batch{batch_idx:06d}_layer{layer_idx:02d}_semantic.pt"
            if sem_file.exists():
                sem = torch.load(sem_file, map_location='cpu')
                # Extract the specific sample from batch
                sample_in_batch = idx % batch_size
                if sample_in_batch < sem.shape[0]:
                    teacher_semantics.append(sem[sample_in_batch])
                else:
                    # Fallback: use first sample (shouldn't happen normally)
                    teacher_semantics.append(sem[0])
            else:
                # If file doesn't exist, use zeros (shouldn't happen)
                print(f"[Warning] Missing semantic cache: {sem_file}")
                teacher_semantics.append(torch.zeros(512))  # Default semantic_dim=512
            
            # Load action predictions
            act_file = self.cache_dir / f"batch{batch_idx:06d}_layer{layer_idx:02d}_action.pt"
            if act_file.exists():
                act = torch.load(act_file, map_location='cpu')
                sample_in_batch = idx % batch_size
                if sample_in_batch < act.shape[0]:
                    teacher_actions.append(act[sample_in_batch])
                else:
                    teacher_actions.append(act[0])
            else:
                print(f"[Warning] Missing action cache: {act_file}")
                teacher_actions.append(torch.zeros(7))  # 7-DoF action
        
        # Add teacher outputs to sample
        sample['teacher_semantics'] = teacher_semantics
        sample['teacher_actions'] = teacher_actions
        
        return sample
