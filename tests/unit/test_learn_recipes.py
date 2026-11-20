"""Every shipped train recipe must load and match a configured reader (P10.2)."""

from __future__ import annotations

from pathlib import Path

import pytest

from torus.config import load_config
from torus.learn.dataset import LearnError
from torus.learn.recipe import load_recipe, load_recipes

TRAIN_DIR = Path(__file__).resolve().parents[2] / "config" / "train"
EXPECTED = {"char_cell", "trocr", "parseq", "vlm_lora_doc", "vlm_lora_qwen3vl"}


def test_all_recipes_load() -> None:
    assert set(load_recipes(TRAIN_DIR)) == EXPECTED


def test_recipe_readers_exist_in_readers_yaml() -> None:
    names = {r.name for r in load_config().readers.readers}
    for recipe in load_recipes(TRAIN_DIR).values():
        assert recipe.reader in names, recipe.name


def test_lora_iff_lora_method() -> None:
    for recipe in load_recipes(TRAIN_DIR).values():
        assert (recipe.method == "lora") == (recipe.lora is not None)


def test_bad_recipe_is_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: x\nunknown_key: 1\n", encoding="utf-8")
    with pytest.raises(LearnError):
        load_recipe(bad)
