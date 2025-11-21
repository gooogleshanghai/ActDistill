from pathlib import Path
from typing import Tuple, Type

from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.models.backbones.vision import ImageTransform
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets import BridgeBatchTransform, BridgeDataset


def get_vla_dataset_and_collator(
    data_root_dir: Path,
    data_mix: str,
    image_transform: ImageTransform,
    tokenizer: PreTrainedTokenizerBase,
    prompt_builder_fn: Type[PromptBuilder],
    default_image_resolution: Tuple[int, int, int],
    padding_side: str = "right",
    predict_stop_token: bool = True,
    shuffle_buffer_size: int = 100_000,
    train: bool = True,
    episodic: bool = False,
    image_aug: bool = False,
    future_action_window_size: int = 0,
    past_action_window_size: int = 1,
    load_all_data_for_training: bool = True,
    base_action_tokenizer: PreTrainedTokenizerBase = None
) -> Tuple[Dataset, ActionTokenizer, PaddedCollatorForActionPrediction]:
    if base_action_tokenizer is None:
        action_tokenizer = None
    else:
        action_tokenizer = ActionTokenizer(base_action_tokenizer)
    batch_transform = BridgeBatchTransform(
        action_tokenizer, tokenizer, image_transform, prompt_builder_fn, predict_stop_token=predict_stop_token
    )
    collator = PaddedCollatorForActionPrediction(
        tokenizer.model_max_length, tokenizer.pad_token_id, padding_side=padding_side
    )

    dataset = BridgeDataset(
        data_root_dir,
        data_mix,
        batch_transform,
        future_action_window_size=future_action_window_size,
        past_action_window_size=past_action_window_size,
    )

    return dataset, action_tokenizer, collator
