import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Type

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.models.backbones.vision import ImageTransform
from prismatic.vla.action_tokenizer import ActionTokenizer

IGNORE_INDEX = -100


def _read_manifest(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"未找到 manifest: {path}")
    content = path.read_text().strip()
    if not content:
        raise ValueError(f"manifest 为空: {path}")
    if content.lstrip().startswith("["):
        data = json.loads(content)
        if not isinstance(data, list):
            raise ValueError(f"manifest 必须是 list: {path}")
        return data
    entries = []
    with open(path, "r") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))
    if not entries:
        raise ValueError(f"manifest 无有效样本: {path}")
    return entries


@dataclass
class BridgeBatchTransform:
    action_tokenizer: ActionTokenizer
    base_tokenizer: PreTrainedTokenizerBase
    image_transform: ImageTransform
    prompt_builder_fn: Type[PromptBuilder]
    predict_stop_token: bool = True

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        with Image.open(Path(sample["image_path"])) as img:
            rgb = img.convert("RGB")
        lang = sample["language_instruction"].lower()

        prompt = self.prompt_builder_fn("openvla")
        if self.action_tokenizer is None:
            answer = ""
        else:
            answer = self.action_tokenizer(sample["actions"])
        prompt.add_turn("human", f"What action should the robot take to {lang}?")
        prompt.add_turn("gpt", answer)

        tokens = self.base_tokenizer(prompt.get_prompt(), add_special_tokens=True).input_ids
        input_ids = torch.tensor(tokens)
        labels = torch.tensor(tokens)

        pixel_values = self.image_transform(rgb)

        action = torch.tensor(sample["actions"], dtype=torch.float32)
        mask = torch.tensor(sample.get("action_mask", [True] * action.shape[0]), dtype=torch.bool)

        if self.action_tokenizer is None:
            labels[:-1] = IGNORE_INDEX
        else:
            labels[: -(len(action) + 1)] = IGNORE_INDEX
        if not self.predict_stop_token:
            labels[-1] = IGNORE_INDEX

        dataset_name = sample.get("dataset_name", "bridge").encode("utf-8")

        return dict(
            pixel_values=pixel_values,
            input_ids=input_ids,
            labels=labels,
            dataset_name=dataset_name,
            actions=action,
            action_masks=mask,
        )


class BridgeDataset(Dataset):
    def __init__(
        self,
        data_root_dir: Path,
        data_mix: str,
        batch_transform: BridgeBatchTransform,
        manifest_filename: str = "manifest.jsonl",
        future_action_window_size: int = 0,
        past_action_window_size: int = 0,
    ) -> None:
        self.batch_transform = batch_transform
        self.data_mix = data_mix
        dataset_dir = Path(data_root_dir) / data_mix
        entries = _read_manifest(dataset_dir / manifest_filename)

        self.samples: List[Dict[str, Any]] = []
        for entry in entries:
            item = dict(entry)
            img_path = Path(item["image_path"])
            if not img_path.is_absolute():
                img_path = dataset_dir / img_path
            item["image_path"] = img_path
            if "language_instruction" not in item:
                raise ValueError("样本缺少 language_instruction")
            if "actions" not in item:
                raise ValueError("样本缺少 actions")
            self.samples.append(item)

        stats_path = dataset_dir / "stats.json"
        if stats_path.exists():
            self.dataset_statistics = json.loads(stats_path.read_text())
        else:
            self.dataset_statistics = self._compute_stats()

    def _compute_stats(self) -> Dict[str, Any]:
        acts = []
        for sample in self.samples:
            arr = np.asarray(sample["actions"], dtype=np.float32)
            arr = arr.reshape(-1, arr.shape[-1])
            acts.append(arr)
        if not acts:
            return {}
        concat = np.concatenate(acts, axis=0)
        stats = {
            "action": {
                "q01": np.quantile(concat, 0.01, axis=0).tolist(),
                "q99": np.quantile(concat, 0.99, axis=0).tolist(),
            }
        }
        return {self.data_mix: stats}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.batch_transform(self.samples[idx])


class DummyDataset(Dataset):
    def __init__(self, samples: Sequence[Dict[str, Any]], batch_transform: BridgeBatchTransform) -> None:
        self.samples = list(samples)
        self.batch_transform = batch_transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.batch_transform(self.samples[idx])
