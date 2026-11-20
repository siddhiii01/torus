# Synthetic fixture sources and licences

Generators under this directory draw **synthetic** pages only. They must not
embed customer documents, real IDs, or private fonts.

## Fonts (optional)

Place downloaded font files in `tests/fixtures/synth/fonts/`. Font binaries are
**not** required for CI: generators fall back to system fonts, then Pillow's
built-in bitmap font.

| File (optional) | Project | Licence | Use |
|---|---|---|---|
| `LiberationSans-Regular.ttf` | [Liberation Fonts](https://github.com/liberationfonts/liberation-fonts) | SIL Open Font Licence 1.1 | Printed form / comb / checkbox / table labels |
| `DejaVuSans.ttf` | [DejaVu Fonts](https://dejavu-fonts.github.io/) | Bitstream Vera + public domain changes | Printed fallback |
| `Caveat-Regular.ttf` | [Caveat](https://fonts.google.com/specimen/Caveat) (Google Fonts) | SIL Open Font Licence 1.1 | Handwriting-like text |
| `PatrickHand-Regular.ttf` | [Patrick Hand](https://fonts.google.com/specimen/Patrick+Hand) (Google Fonts) | SIL Open Font Licence 1.1 | Handwriting-like text |

Do **not** commit font files that you are not allowed to redistribute. Prefer
OFL fonts listed above.

## Datasets

No external handwriting datasets are **vendored** in the repo. Stroke jitter is
applied in code (`generators/handwriting.py`). Single-cell glyphs for the
character-cell reader are drawn in `generators/char_cell.py` (and the training
helper `torus.readers.char_cell.train_data.synth_glyph`).

Optional **download** for training the cell classifier (P5.5), never committed:

| Dataset | How | Licence | Use |
|---|---|---|---|
| [EMNIST ByClass](https://www.nist.gov/itl/products-and-services/emnist-dataset) | `python -m torus.readers.char_cell train --source emnist` (needs `.[train]` / HuggingFace `datasets`) | NIST / CC0-style redistribution of NIST Special Database | Digits + letters for the cell MLP |

Cache location: `$TORUS_MODEL_DIR/cache/emnist` (or `$TORUS_DATA_DIR` /
`$HF_HOME`), outside the git checkout.

## Generated artefacts

- Images returned by generators are **in-memory** (`PIL.Image`).
- Do not commit `*.png` / `*.jpg` under this tree (repo `.gitignore` already
  blocks common image extensions).
- A local `generated/` cache directory, if created for debugging, must stay
  untracked.
