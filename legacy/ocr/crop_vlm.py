"""Crop-level reader escalation (TrOCR / Florence-2) for known digit_crop FNs.

Architecture: locate stays in deb_extract; specialized digit readers run first;
this module only re-reads located crops that came back empty / disagreed.
Callers still own fill-empties-only and must run the same schema validators.

No full-page OCR. No external document APIs. Qwen-family is intentionally absent.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

import numpy as np

from ocr.digit_crop import (
    DigitRead,
    black_ink_mask,
    crop_norm,
    digits_only,
    pick_validated,
    prep_digit_crop,
    sliding,
)
from ocr.pin_master import (
    validate_city_token as _pin_validate_city,
    validate_state_token as _pin_validate_state,
)

# Lazy HF stacks — import stays cheap until escalate is requested.
_models: dict[str, object] = {}

READER_MODELS = {
    "trocr_handwritten": "microsoft/trocr-base-handwritten",
    "trocr_printed": "microsoft/trocr-base-printed",
    "florence2_base": "microsoft/Florence-2-base",
}

DEFAULT_LADDER = ("trocr_handwritten", "trocr_printed")


@dataclass(frozen=True)
class TextRead:
    field: str
    value: str
    conf: float
    reader: str
    bbox: tuple[float, float, float, float]


def crop_vlm_available(reader: str = "trocr_handwritten") -> tuple[bool, str]:
    """Return (ok, reason). Soft-check only — does not download weights."""
    if reader not in READER_MODELS and reader not in {"trocr", "florence"}:
        return False, f"unknown reader {reader}"
    try:
        import transformers  # noqa: F401
    except ImportError:
        return False, "install transformers (pip install transformers) for crop VLM escalate"
    try:
        import torch  # noqa: F401
    except ImportError:
        return False, "torch required for crop VLM escalate"
    return True, "ok"


def _device():
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _load_trocr(model_id: str):
    key = f"trocr:{model_id}"
    if key not in _models:
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        processor = TrOCRProcessor.from_pretrained(model_id)
        model = VisionEncoderDecoderModel.from_pretrained(model_id)
        model.to(_device())
        model.eval()
        _models[key] = (processor, model)
    return _models[key]


def _load_florence(model_id: str):
    key = f"florence:{model_id}"
    if key not in _models:
        from transformers import AutoModelForCausalLM, AutoProcessor

        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
        model.to(_device())
        model.eval()
        _models[key] = (processor, model)
    return _models[key]


def _pil_rgb(img: np.ndarray):
    from PIL import Image

    if img.size == 0:
        return None
    if img.ndim == 2:
        return Image.fromarray(img).convert("RGB")
    return Image.fromarray(img).convert("RGB")


def recognize_trocr(img: np.ndarray, *, variant: str = "handwritten") -> tuple[str, float]:
    """Return (text, conf). conf is heuristic — TrOCR does not expose token probs cheaply."""
    if img.size == 0:
        return "", 0.0
    model_id = READER_MODELS["trocr_handwritten" if variant == "handwritten" else "trocr_printed"]
    processor, model = _load_trocr(model_id)
    import torch

    pil = _pil_rgb(img)
    if pil is None:
        return "", 0.0
    pixel_values = processor(images=pil, return_tensors="pt").pixel_values.to(_device())
    with torch.no_grad():
        generated = model.generate(pixel_values, max_new_tokens=32)
    text = processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
    conf = min(0.92, 0.55 + 0.03 * len(re.sub(r"\s+", "", text))) if text else 0.0
    return text, conf


def recognize_florence(img: np.ndarray) -> tuple[str, float]:
    """Florence-2 OCR task on a tight crop. Local only."""
    if img.size == 0:
        return "", 0.0
    processor, model = _load_florence(READER_MODELS["florence2_base"])
    import torch

    pil = _pil_rgb(img)
    if pil is None:
        return "", 0.0
    task = "<OCR>"
    inputs = processor(text=task, images=pil, return_tensors="pt")
    inputs = {k: v.to(_device()) if hasattr(v, "to") else v for k, v in inputs.items()}
    with torch.no_grad():
        generated = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=64,
            num_beams=1,
            do_sample=False,
        )
    raw = processor.batch_decode(generated, skip_special_tokens=False)[0]
    parsed = processor.post_process_generation(raw, task=task, image_size=pil.size)
    text = ""
    if isinstance(parsed, dict):
        text = str(parsed.get(task) or parsed.get("<OCR>") or "").strip()
    elif isinstance(parsed, str):
        text = parsed.strip()
    conf = min(0.9, 0.5 + 0.03 * len(re.sub(r"\s+", "", text))) if text else 0.0
    return text, conf


def collect_vlm_digit_votes(
    crop: np.ndarray,
    *,
    readers: tuple[str, ...] = DEFAULT_LADDER,
    scale: float = 3.0,
) -> list[tuple[str, str, float]]:
    """(reader, digit_string, conf) from escalate ladder on ink-separated crop variants."""
    if crop.size == 0:
        return []
    ok, reason = crop_vlm_available(readers[0] if readers else "trocr_handwritten")
    if not ok:
        return []

    votes: list[tuple[str, str, float]] = []
    variants = [
        ("prep", prep_digit_crop(crop, scale=scale, trim=True)),
        ("full", prep_digit_crop(crop, scale=scale, trim=False)),
        ("raw_x2", prep_digit_crop(crop, scale=2.0, trim=False)),
    ]
    for vname, prepped in variants:
        if prepped.size == 0:
            continue
        for reader in readers:
            tag = f"{reader}_{vname}"
            try:
                if reader in {"trocr_handwritten", "trocr"}:
                    text, conf = recognize_trocr(prepped, variant="handwritten")
                elif reader == "trocr_printed":
                    text, conf = recognize_trocr(prepped, variant="printed")
                elif reader in {"florence2_base", "florence"}:
                    # Florence once on prep only — heavier than TrOCR.
                    if vname != "prep":
                        continue
                    text, conf = recognize_florence(prepped)
                else:
                    continue
            except Exception:
                continue
            votes.append((tag, digits_only(text), conf))
    return votes


def _prior_digit_strings(prior_votes: list[tuple[str, str, float]] | None) -> list[str]:
    if not prior_votes:
        return []
    return [d for _r, d, _c in prior_votes if d]


def corroborates_digits(
    value: str,
    prior_votes: list[tuple[str, str, float]] | None,
    *,
    lengths: list[int],
) -> bool:
    """Anti-hallucination: if specialized readers saw digits, VLM must echo a window.

    When prior votes are empty (readers totally blank), validators alone gate the hit.
    """
    min_n = min(lengths) if lengths else 1
    # Ignore tiny OCR crumbs — only echo-check when a specialized reader saw a
    # digit run near the expected length (disagree path). Blank priors → validators only.
    priors = [d for d in _prior_digit_strings(prior_votes) if len(d) >= max(2, min_n - 2)]
    if not priors:
        return True
    digs = re.sub(r"\D", "", value)
    if not digs:
        return False
    # DOB validator returns ISO; crop votes are usually ddmmyyyy — check both.
    forms = [digs]
    if re.match(r"^\d{4}-\d{2}-\d{2}$", value):
        yyyy, mm, dd = value.split("-")
        forms.append(f"{dd}{mm}{yyyy}")
    for form in forms:
        for n in lengths:
            windows = sliding(form, n) if len(form) >= n else ([form] if form else [])
            for window in windows:
                for prior in priors:
                    if window and window in prior:
                        return True
                    if prior and prior in window:
                        return True
                    for pwin in sliding(prior, n):
                        if pwin == window:
                            return True
                        # allow 1-char disagreement on long codes
                        if len(pwin) == len(window) == n and n >= 6:
                            diff = sum(a != b for a, b in zip(pwin, window))
                            if diff <= 1:
                                return True
    return False


def escalate_digit_field(
    page_img: np.ndarray,
    bbox: tuple[float, float, float, float],
    *,
    field: str,
    lengths: list[int],
    validator: Callable[[str], str | None],
    readers: tuple[str, ...] = DEFAULT_LADDER,
    prefer_len: int | None = None,
    prior_votes: list[tuple[str, str, float]] | None = None,
    require_prior_echo: bool = True,
) -> DigitRead | None:
    """Run escalate ladder; return only schema-valid (and optionally corroborated) hits."""
    crop = crop_norm(page_img, bbox)
    if crop.size == 0:
        return None
    votes = collect_vlm_digit_votes(crop, readers=readers)
    if not votes:
        return None
    value, reader, conf = pick_validated(
        votes,
        lengths,
        validator,
        prefer_len=prefer_len,
        require_agreement=False,
        min_support=1,
    )
    if value is None or reader is None:
        return None
    if require_prior_echo and not corroborates_digits(value, prior_votes, lengths=lengths):
        return None
    return DigitRead(field=field, value=value, conf=conf, reader=f"vlm:{reader}", bbox=bbox)


_EMAIL_RE = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.I)

# Blank comb / grid-only crops: non-grid dark-pixel density below this → skip VLM.
DEFAULT_MIN_INK_DENSITY = 0.004
DEFAULT_MIN_INK_PIXELS = 18


def ink_density(img: np.ndarray) -> tuple[float, int]:
    """Fraction and count of non-grid dark ink pixels (reuses digit_crop black_ink_mask)."""
    if img.size == 0:
        return 0.0, 0
    masked = black_ink_mask(img)
    gray = masked[:, :, 0] if masked.ndim == 3 else masked
    ink = gray < 128
    count = int(ink.sum())
    dens = float(count) / float(ink.size) if ink.size else 0.0
    return dens, count


def has_handwriting_ink(
    img: np.ndarray,
    *,
    min_density: float = DEFAULT_MIN_INK_DENSITY,
    min_pixels: int = DEFAULT_MIN_INK_PIXELS,
) -> bool:
    """True when the crop has enough non-grid ink to justify a free-text VLM call."""
    dens, count = ink_density(img)
    return dens >= min_density and count >= min_pixels


def validate_email(text: str) -> str | None:
    m = _EMAIL_RE.search(text.replace(" ", ""))
    if not m:
        # common OCR glue: namegmail.com
        m2 = re.search(r"([A-Z0-9._%+-]{3,})gmail\.com", text.replace(" ", ""), re.I)
        if m2:
            return f"{m2.group(1)}@gmail.com"
        return None
    return m.group(0)


def validate_city_token(text: str, *, pin: str | None = None) -> str | None:
    """Strict vocab gate (PIN-scoped when pin known). Replaces length-only alpha check."""
    return _pin_validate_city(text, pin=pin)


def validate_state_token(text: str) -> str | None:
    """Closed Indian state list — exact / near-exact after normalize."""
    return _pin_validate_state(text)


def recognize_crop_text(
    page_img: np.ndarray,
    bbox: tuple[float, float, float, float],
    *,
    readers: tuple[str, ...] = DEFAULT_LADDER,
    require_ink: bool = True,
    min_ink_density: float = DEFAULT_MIN_INK_DENSITY,
    min_ink_pixels: int = DEFAULT_MIN_INK_PIXELS,
) -> list[tuple[str, str, float]]:
    """Free-text votes for email / city / state field crops.

    Blank / grid-only combs are skipped before any VLM call (kills I/L/1 inventions).
    """
    crop = crop_norm(page_img, bbox)
    if crop.size == 0:
        return []
    if require_ink and not has_handwriting_ink(
        crop, min_density=min_ink_density, min_pixels=min_ink_pixels
    ):
        return []
    ok, _ = crop_vlm_available(readers[0] if readers else "trocr_handwritten")
    if not ok:
        return []
    from ocr.digit_crop import upscale

    variants = [("raw", upscale(crop, 2.0)), ("x3", upscale(crop, 3.0))]
    votes: list[tuple[str, str, float]] = []
    for vname, img in variants:
        if img.size == 0:
            continue
        for reader in readers:
            try:
                if reader in {"trocr_handwritten", "trocr"}:
                    text, conf = recognize_trocr(img, variant="handwritten")
                elif reader == "trocr_printed":
                    text, conf = recognize_trocr(img, variant="printed")
                elif reader in {"florence2_base", "florence"}:
                    if vname != "raw":
                        continue
                    text, conf = recognize_florence(img)
                else:
                    continue
            except Exception:
                continue
            if text:
                votes.append((f"{reader}_{vname}", text, conf))
    return votes


def escalate_text_field(
    page_img: np.ndarray,
    bbox: tuple[float, float, float, float],
    *,
    field: str,
    validator: Callable[[str], str | None],
    readers: tuple[str, ...] = DEFAULT_LADDER,
    require_ink: bool = True,
) -> TextRead | None:
    votes = recognize_crop_text(page_img, bbox, readers=readers, require_ink=require_ink)
    best: tuple[str, str, float] | None = None
    for reader, text, conf in votes:
        val = validator(text)
        if not val:
            continue
        if best is None or conf > best[2]:
            best = (reader, val, conf)
    if best is None:
        return None
    reader, value, conf = best
    return TextRead(field=field, value=value, conf=conf, reader=f"vlm:{reader}", bbox=bbox)
