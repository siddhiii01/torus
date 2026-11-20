"""Seeded synthetic page generators for unit and component tests.

Generated images stay in memory. Do not commit page images or OCR crops.
See ``SOURCES.md`` for public font licences.
"""

from __future__ import annotations

from tests.fixtures.synth.generators import (
    generate,
    generate_blank,
    generate_bordered,
    generate_char_cell,
    generate_checkbox,
    generate_checkbox_group,
    generate_comb_row,
    generate_faded,
    generate_fax_degraded,
    generate_handwriting,
    generate_handwriting_page,
    generate_label_anchor_page,
    generate_layout_page,
    generate_near_blank,
    generate_perspective_photo,
    generate_printed_form,
    generate_rotated,
    generate_signature,
    generate_skewed,
    generate_split_table,
    generate_stamp,
    generate_table,
    write_synth_jpeg_with_exif_orientation,
    write_synth_multipage_tiff,
    write_synth_pdf_image_only,
    write_synth_pdf_multipage,
    write_synth_pdf_with_text,
    write_synth_png,
)
from tests.fixtures.synth.testset import (
    MINI_SET_ID,
    MINI_TEMPLATE_ID,
    build_mini_testset,
    write_synth_pdf,
)
from tests.fixtures.synth.types import RegionTruth, SynthKind, SynthPage

__all__ = [
    "MINI_SET_ID",
    "MINI_TEMPLATE_ID",
    "RegionTruth",
    "SynthKind",
    "SynthPage",
    "build_mini_testset",
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
    "write_synth_pdf",
    "write_synth_pdf_image_only",
    "write_synth_pdf_multipage",
    "write_synth_pdf_with_text",
    "write_synth_png",
]
