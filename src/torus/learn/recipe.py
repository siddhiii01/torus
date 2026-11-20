"""Fine-tune recipes (P10.2): typed schema + loader for ``config/train/*.yaml``.

A recipe states *what* to train (base model, dataset kind, hyperparameters). It
holds no document data and triggers no download; weights and datasets live under
``TORUS_MODEL_DIR`` / ``TORUS_DATA_DIR``. Running a recipe is a later step (S7).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from torus.learn.dataset import LearnError, SampleKind


class LoraSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    r: int = Field(gt=0)
    alpha: int = Field(gt=0)
    dropout: float = Field(ge=0.0, lt=1.0)
    target_modules: list[str] = Field(min_length=1)


class TrainRecipe(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    reader: str = Field(min_length=1, description="Reader name in readers.yaml.")
    method: Literal["full", "lora"]
    dataset_kind: SampleKind
    base_model: str = Field(min_length=1)
    base_model_source: Literal["model_dir", "hf_cache", "none"]
    framework: Literal["numpy", "torch"]
    install_extra: str | None = None
    seed: int = 0
    epochs: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    learning_rate: float = Field(gt=0)
    weight_decay: float = Field(ge=0.0, default=0.0)
    max_target_length: int | None = Field(default=None, gt=0)
    image_size: list[int] | None = None
    min_vram_gb: float = Field(ge=0.0, default=0.0)
    min_ram_gb: float = Field(ge=0.0, default=1.0)
    lora: LoraSpec | None = None
    # Output goes to the registry (P10.3) under TORUS_MODEL_DIR, never git.
    output_subdir: str = Field(min_length=1)
    notes: str = ""

    def model_post_init(self, _ctx: object) -> None:
        if (self.method == "lora") != (self.lora is not None):
            raise ValueError(
                "lora settings are required for, and only for, method=lora"
            )
        if self.image_size is not None and (
            len(self.image_size) != 2 or min(self.image_size) <= 0
        ):
            raise ValueError("image_size must be [height, width] with positive values")


def load_recipe(path: Path) -> TrainRecipe:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        return TrainRecipe.model_validate(raw)
    except (OSError, yaml.YAMLError, ValidationError, ValueError) as exc:
        raise LearnError(f"invalid train recipe: {path.name}") from exc


def load_recipes(train_dir: Path) -> dict[str, TrainRecipe]:
    recipes: dict[str, TrainRecipe] = {}
    for path in sorted(train_dir.glob("*.yaml")):
        recipe = load_recipe(path)
        if recipe.name in recipes:
            raise LearnError(f"duplicate recipe name in {path.name}")
        recipes[recipe.name] = recipe
    return recipes
