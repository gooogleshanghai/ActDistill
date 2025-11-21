"""
train.py

Training script for Vision-Language-Action (VLA) Policies, built on top of pretrained VLMs, trained using mixtures of
the Open-X Embodiment dataset. Performs training in native PyTorch, using Fully-Sharded Data Parallel (FSDP) to run
distributed across GPUs (and nodes). By default, assumes that CUDA toolkit is >= 11.0 (to support BF16 mixed precision).

Notes & Prerequisites:
    - If you want to set a custom location for all HF / TIMM artifacts --> `export HF_HOME="<PATH>"` *before* running!
        => For example (add to end of .bashrc): `export HF_HOME="/mnt/fsx/skaramcheti/cache"`
    - If you want to suppress random Tensorflow logs --> `export TF_CPP_MIN_LOG_LEVEL=3`

Run with:
    - [Single Node One-GPU (Debug)] : torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/train.py
    - [Single Node Multi-GPU (= $K)]: torchrun --standalone --nnodes 1 --nproc-per-node $K vla-scripts/train.py
"""

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, Union

import draccus
import torch
import torch.distributed as dist
import yaml
import wandb

from prismatic.overwatch import initialize_overwatch
from prismatic.util import set_global_seed
from prismatic.vla import get_vla_dataset_and_collator
from prismatic.vla.datasets.utils import save_dataset_statistics

from training import VLAMetrics, get_train_strategy
from conf import VLAConfig, VLARegistry
from vla import load, load_vla
from vla import CogACT
from prismatic.vla.datasets.cached_teacher_dataset import CachedTeacherDataset

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# RANDOM_SEED and other environment variables with defaults
random_seed = int(os.environ.get('RANDOM_SEED', '42'))
print(f" random_seed: {random_seed} {type(random_seed)}")
# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


@dataclass
class TrainConfig:
    # fmt: off

    # VLAConfig (`conf/vla.py`); override with --vla.type `VLARegistry.<VLA>.vla_id`
    vla: VLAConfig = field(
        default_factory=VLAConfig.get_choice_class(VLARegistry.EXP_COGACT_OXE_MAGIC_SOUP_PLUS_MINUS.vla_id)
    )

    # Directory Paths
    data_root_dir: Path = Path(                                     # Path to Open-X dataset directory
        "datasets/open-x-embodiment"
    )
    run_root_dir: Path = Path("runs")                               # Path to directory to store logs & checkpoints

    # Resume Run Parameters
    pretrained_checkpoint: Optional[Union[str, Path]] = None                  # Absolute Path to Checkpoint
    is_resume: bool = True                                          # Whether we are continuing a prior training run
                                                                    # (only applicable given pretrained checkpoint)
    resume_step: Optional[int] = None                               # Global Step to Resume (should match checkpoint)
    resume_epoch: Optional[int] = None                              # Epoch to Resume (should match checkpoint)

    # Run Arguments
    run_id: Optional[str] = None                                    # Run ID for logging, Weights & Biases
    run_id_note: Optional[str] = None                               # Extra note for logging, Weights & Biases
    save_interval: int = 2500                                       # Interval for saving checkpoints (in steps)
    image_aug: bool = False                                         # Whether to enable image augmentations
    # seed: int = 42                                                  # Random seed (for reproducibility)
    seed: int = random_seed                                                 # Random seed (for reproducibility)

    # HF Hub Credentials (for any gated models)
    hf_token: Union[str, Path] = Path(".hf_token")                  # Environment variable or Path to HF Token

    # Tracking Parameters
    # trackers: Tuple[str, ...] = ("wandb")                  # Trackers to initialize (if W&B, add config!)
    trackers: Tuple[str, ...] = ("jsonl")                         # Trackers to initialize (if W&B, add config!)
    wandb_project: str = ""                                         # Name of W&B project to log to (use default!)
    wandb_entity: str = ""                                          # Name of entity to log under
    repeated_diffusion_steps: int = 8                               # Repeated steps for training action model (a diffusion model)
    load_all_data_for_training: bool = True                         # Load all training data 
    future_action_window_size: int = 15                             # Action chunking, predicting future actions + current action
    past_action_window_size: int = 0                                # Action history window size, not used now, set to be 0 
    action_model_type: str = 'DiT-B'                                # Action model type, chose from ['DiT-S', 'DiT-B', 'DiT-L']
    action_dim: int = 7                                             # Dimension of action space
    load_dit: bool = True
    teacher_cache_dir: Optional[Path] = None                        # Directory containing cached teacher outputs

    def __post_init__(self) -> None:
        """Lift optimization parameters from `self.vla` for ease of use =>> validate on `expected_world_size`"""
        self.epochs = self.vla.epochs
        self.max_steps = self.vla.max_steps
        self.global_batch_size = self.vla.global_batch_size
        self.per_device_batch_size = self.vla.per_device_batch_size

        self.learning_rate = self.vla.learning_rate
        self.weight_decay = self.vla.weight_decay
        self.max_grad_norm = self.vla.max_grad_norm
        self.lr_scheduler_type = self.vla.lr_scheduler_type
        self.warmup_ratio = self.vla.warmup_ratio

        self.train_strategy = self.vla.train_strategy

        # [Validate] Assert on `expected_world_size`
        assert (
            self.vla.expected_world_size == overwatch.world_size()
        ), f"Expected World Size = {self.vla.expected_world_size} but Found {overwatch.world_size()} GPUs!"

    # fmt: on


@draccus.wrap()
def train(cfg: TrainConfig) -> None:
    torch.cuda.set_device(device_id := overwatch.local_rank())
    torch.cuda.empty_cache()

    # Configure Unique Run Name & Save Directory
    vla_id = cfg.vla.vla_id
    cfg.run_id = (
        f"{vla_id}+n{cfg.vla.expected_world_size // 8}+b{cfg.per_device_batch_size}+x{cfg.seed}"
        if cfg.run_id is None
        else cfg.run_id
    )
    if cfg.run_id_note is not None:
        cfg.run_id += f"--{cfg.run_id_note}"
    if cfg.image_aug:
        cfg.run_id += "--image_aug"

    if isinstance(cfg.hf_token, Path):
        hf_token = cfg.hf_token.read_text().strip()
    else:
        hf_token = os.environ.get(str(cfg.hf_token), "")
    worker_init_fn = set_global_seed(cfg.seed, get_worker_init_fn=True)
    os.makedirs(run_dir := (cfg.run_root_dir / cfg.run_id), exist_ok=True)
    os.makedirs(cfg.run_root_dir / cfg.run_id / "checkpoints", exist_ok=True)

    # Save Configuration =>> additionally save a JSON version for later HF Integration
    if overwatch.is_rank_zero():
        draccus.dump(cfg, open(run_dir / "config.yaml", "w"))
        with open(run_dir / "config.yaml", "r") as f_yaml, open(run_dir / "config.json", "w") as f_json:
            yaml_cfg = yaml.safe_load(f_yaml)
            json.dump(yaml_cfg, f_json, indent=2)

    dist.barrier()
    if cfg.pretrained_checkpoint is not None:
        if cfg.is_resume:
            assert int(re.search("step-(.+?)-", cfg.pretrained_checkpoint.name).group(1)) == cfg.resume_step
            assert int(re.search("epoch-(.+?)-", cfg.pretrained_checkpoint.name).group(1)) == cfg.resume_epoch
        vla = load_vla(
            cfg.pretrained_checkpoint,
            hf_token=hf_token,
            load_for_training=True,
            action_model_type=cfg.action_model_type,
            action_dim=cfg.action_dim,
            future_action_window_size=cfg.future_action_window_size,
            past_action_window_size=cfg.past_action_window_size,
            load_dit=cfg.load_dit,
        )
    else:
        vlm = load(cfg.vla.base_vlm, hf_token=hf_token, load_for_training=True)
        vla = CogACT(
            vlm,
            action_model_type=cfg.action_model_type,
            action_dim=cfg.action_dim,
            future_action_window_size=cfg.future_action_window_size,
            past_action_window_size=cfg.past_action_window_size,
            load_dit=cfg.load_dit,
        )
        del vlm

    for param in vla.parameters():
        assert param.dtype == torch.float32, f"Loaded VLM parameter not in full precision: {param}"

    if not cfg.vla.freeze_vision_backbone and not cfg.vla.freeze_llm_backbone:
        stage = "full-finetune"  # Full fine-tuning
    elif cfg.vla.freeze_vision_backbone and not cfg.vla.freeze_llm_backbone:
        stage = "finetune"  # Frozen vision encoder
    elif cfg.vla.freeze_vision_backbone and cfg.vla.freeze_llm_backbone:
        stage = "align"  # Fine-tuning projector
    elif not cfg.vla.freeze_vision_backbone and cfg.vla.freeze_llm_backbone and cfg.vla.unfreeze_last_llm_layer:
        stage = "vla-sandwich-train"  # Fine-tuning vision encoder, projector, and LLM last layer
    elif cfg.vla.freeze_vision_backbone and cfg.vla.freeze_llm_backbone and cfg.vla.unfreeze_last_llm_layer:
        stage = "vla-last-layer-train"  # Fine-tuning LLM last layer only
    else:
        raise ValueError(
            "Weight freezing configuration not supported. VLA config has the following parameters: "
            f"freeze_vision_backbone: {cfg.vla.freeze_vision_backbone}"
            f"freeze_llm_backbone: {cfg.vla.freeze_llm_backbone}"
            f"unfreeze_last_llm_layer: {cfg.vla.unfreeze_last_llm_layer}"
        )

    vla.freeze_backbones(stage)

    if cfg.vla.freeze_vision_backbone and cfg.vla.freeze_llm_backbone:
        for name, param in vla.vlm.named_parameters():
            param.requires_grad = False

    num_params = sum(p.numel() for p in vla.parameters())
    num_trainable_params = sum(p.numel() for p in vla.parameters() if p.requires_grad)

    vla_dataset, _, collator = get_vla_dataset_and_collator(
        cfg.data_root_dir,
        cfg.vla.data_mix,
        image_transform=vla.vision_backbone.get_image_transform(),
        tokenizer=vla.llm_backbone.get_tokenizer(),
        prompt_builder_fn=vla.llm_backbone.prompt_builder_fn,
        default_image_resolution=vla.vision_backbone.default_image_resolution,
        shuffle_buffer_size=cfg.vla.shuffle_buffer_size,
        image_aug=cfg.image_aug,
        load_all_data_for_training=cfg.load_all_data_for_training,
        future_action_window_size=cfg.future_action_window_size,
        past_action_window_size=cfg.past_action_window_size,
    )
    if cfg.teacher_cache_dir is not None:
        cache_dir = Path(cfg.teacher_cache_dir)
        if not cache_dir.exists():
            raise FileNotFoundError(f"Teacher cache directory not found: {cache_dir}")
        num_layers = vla.llm_backbone.llm.config.num_hidden_layers
        vla_dataset = CachedTeacherDataset(
            base_dataset=vla_dataset,
            teacher_cache_dir=str(cache_dir),
            num_layers=num_layers,
        )

        semantic_heads_path = cache_dir.parent / "semantic_heads_final.pth"
        if semantic_heads_path.exists():
            saved_heads = torch.load(semantic_heads_path, map_location='cpu')
            if 'gnn_semantic_heads' in saved_heads:
                vla.llm_backbone.llm.gnn_semantic_heads.load_state_dict(saved_heads['gnn_semantic_heads'])
            if 'action_heads' in saved_heads:
                vla.llm_backbone.llm.action_heads.load_state_dict(saved_heads['action_heads'])
        else:
            print(f"[Warning] Stage 1 semantic heads not found at {semantic_heads_path}, using random initialization")
    if overwatch.is_rank_zero():
        save_dataset_statistics(vla_dataset.dataset_statistics, run_dir)

    dist.barrier()
    train_strategy = get_train_strategy(
        train_strategy=cfg.train_strategy,
        vlm=vla,
        device_id=device_id,
        stage=stage,
        epochs=cfg.epochs,
        max_steps=cfg.max_steps,
        global_batch_size=cfg.global_batch_size,
        per_device_batch_size=cfg.per_device_batch_size,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        max_grad_norm=cfg.max_grad_norm,
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_ratio=cfg.warmup_ratio,
        enable_gradient_checkpointing=cfg.vla.enable_gradient_checkpointing,
        enable_mixed_precision_training=cfg.vla.enable_mixed_precision_training,
        reduce_in_full_precision=cfg.vla.reduce_in_full_precision,
        worker_init_fn=worker_init_fn,
    )
    train_strategy.run_setup(run_dir=run_dir, n_train_examples=len(vla_dataset))
    if cfg.pretrained_checkpoint is not None and cfg.is_resume:
        train_strategy.load_optimizer_and_scheduler(cfg.pretrained_checkpoint)

    metrics = VLAMetrics(
        cfg.trackers,
        cfg.run_id,
        run_dir,
        draccus.encode(cfg),
        wandb_project=cfg.wandb_project,
        wandb_entity=cfg.wandb_entity,
        resume_step=cfg.resume_step,
        resume_epoch=cfg.resume_epoch,
    )

    train_strategy.run_vla_training(
        vla_dataset,
        collator,
        metrics,
        save_interval=cfg.save_interval,
        action_model=True,
    )

   
    metrics.finalize()

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    train()
