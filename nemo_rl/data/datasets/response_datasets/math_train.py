# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
from typing import Any

from datasets import Dataset

from nemo_rl.data.datasets.raw_dataset import RawDataset

# Hendrycks MATH train split (~7.5k), boxed answers, prefetched from a login
# node into a plain JSONL. Loaded via Dataset.from_list (not load_dataset) so
# training needs no HF hub, no builder FileLock, and no os.statvfs -- the
# training container is HF-offline and Lustre statvfs can raise EROFS inside
# the datasets FileLock. Override the path with MATH_TRAIN_JSONL if needed.
DEFAULT_MATH_TRAIN_JSONL = (
    "/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/snorouzi/"
    "eval_data/math_train/train.jsonl"
)


class MathTrainDataset(RawDataset):
    """Hendrycks MATH train split as GRPO prompts (problem -> boxed answer)."""

    def __init__(self, data_path: str | None = None, **kwargs) -> None:
        self.task_name = "MATH"

        path = data_path or os.environ.get(
            "MATH_TRAIN_JSONL", DEFAULT_MATH_TRAIN_JSONL
        )
        with open(path, "r", encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]

        self.dataset = Dataset.from_list(rows)
        self.dataset = self.dataset.map(
            self.format_data,
            remove_columns=self.dataset.column_names,
        )
        self.val_dataset = None

    def format_data(self, data: dict[str, Any]) -> dict[str, Any]:
        return {
            "messages": [
                {"role": "user", "content": data["problem"]},
                {"role": "assistant", "content": data["answer"]},
            ],
            "task_name": self.task_name,
        }
