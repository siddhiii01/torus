"""Seeded synthetic page generators."""

from __future__ import annotations

from tests.fixtures.synth.generators.blank import generate_blank, generate_near_blank
from tests.fixtures.synth.generators.char_cell import generate_char_cell
from tests.fixtures.synth.generators.checkbox import (
    generate_checkbox,
    generate_checkbox_group,
)
from tests.fixtures.synth.generators.comb import generate_comb_row
from tests.fixtures.synth.generators.geometry import (
    generate_bordered,
    generate_perspective_photo,
    generate_skewed,
)
from tests.fixtures.synth.generators.handwriting import (
    generate_handwriting,
    generate_handwriting_page,
)
from tests.fixtures.synth.generators.ingest_files import (
    write_synth_jpeg_with_exif_orientation,
    write_synth_multipage_tiff,
    write_synth_pdf_image_only,
    write_synth_pdf_multipage,
    write_synth_pdf_with_text,
    write_synth_png,
)
from tests.fixtures.synth.generators.label_anchor import generate_label_anchor_page
from tests.fixtures.synth.generators.layout_page import generate_layout_page
from tests.fixtures.synth.generators.printed_form import generate_printed_form
from tests.fixtures.synth.generators.rotation import (
    generate_faded,
    generate_fax_degraded,
    generate_rotated,
)
from tests.fixtures.synth.generators.stamp import generate_signature, generate_stamp
from tests.fixtures.synth.generators.table import generate_split_table, generate_table
from tests.fixtures.synth.types import SynthKind, SynthPage

__all__ = [
    "generate",
    "generate_blank",
    "generate_bordered",
    "generate_char_cell",
    "generate_checkbox",
    "generate_checkbox_group",
    "generate_comb_row",
    "generate_faded",
    "generate_fax_degraded",
    "generate_handwriting",
    "generate_handwriting_page",
    "generate_label_anchor_page",
    "generate_layout_page",
    "generate_near_blank",
    "generate_perspective_photo",
    "generate_printed_form",
    "generate_rotated",
    "generate_signature",
    "generate_skewed",
    "generate_split_table",
    "generate_stamp",
    "generate_table",
    "write_synth_jpeg_with_exif_orientation",
    "write_synth_multipage_tiff",
    "write_synth_pdf_image_only",
    "write_synth_pdf_multipage",
    "write_synth_pdf_with_text",
    "write_synth_png",
]


def generate(kind: SynthKind, seed: int = 0, **kwargs: object) -> SynthPage:
    """Dispatch to a generator by kind name."""
    dispatch = {
        "printed_form": generate_printed_form,
        "comb_row": generate_comb_row,
        "char_cell": generate_char_cell,
        "checkbox": generate_checkbox,
        "handwriting": generate_handwriting,
        "rotated": generate_rotated,
        "fax_degraded": generate_fax_degraded,
        "faded": generate_faded,
        "stamp": generate_stamp,
        "signature": generate_signature,
        "skewed": generate_skewed,
        "perspective_photo": generate_perspective_photo,
        "bordered": generate_bordered,
        "blank": generate_blank,
        "layout_page": generate_layout_page,
        "label_anchor": generate_label_anchor_page,
        "table": generate_table,
    }
    try:
        fn = dispatch[kind]
    except KeyError as exc:
        msg = f"unknown synth kind {kind!r}"
        raise ValueError(msg) from exc
    return fn(seed, **kwargs)  # type: ignore[operator]
