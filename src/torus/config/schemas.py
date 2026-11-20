"""Pydantic schemas for config/*.yaml.

Document and client knowledge lives here as data. Pipeline stages read these
models through ``torus.config.load_config``; they never hard-code field lists.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from torus.core.models import ReaderCaps
from torus.core.types import Charset, FieldStatus, FieldType


class Cardinality(StrEnum):
    """How many output rows a field contributes to."""

    PER_PACK = "per_pack"
    PER_PERSON = "per_person"
    PER_ADDRESS = "per_address"


class DeviceKind(StrEnum):
    """Preferred execution device for a reader."""

    CPU = "cpu"
    CUDA = "cuda"
    AUTO = "auto"


class ConfidenceThresholds(BaseModel):
    """Calibrated cut-offs for status assignment (fitted later on the eval set)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    confirmed_min: float = Field(ge=0.0, le=1.0)
    likely_min: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _likely_not_above_confirmed(self) -> ConfidenceThresholds:
        if self.likely_min > self.confirmed_min:
            msg = (
                "likely_min must be <= confirmed_min, "
                f"got {self.likely_min} > {self.confirmed_min}"
            )
            raise ValueError(msg)
        return self


class VotingRules(BaseModel):
    """Agreement rules before a value may be confirmed."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    critical_min_families: int = Field(ge=1)
    noncritical_min_readers: int = Field(ge=1)
    strong_reading_min_score: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description=(
            "A reading counts as strong when its raw score (or mean per-character "
            "confidence) is at least this. Readings with no score are never strong."
        ),
    )
    noncritical_require_validator: bool = Field(
        default=True,
        description=(
            "Non-critical fields also need at least one validator result and all "
            "validators passing. With no validator the rule is not met (never vacuous)."
        ),
    )


class EscalationTier(BaseModel):
    """One escalation round: which extra views of a disputed region to read."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    readers: list[str] = Field(
        default_factory=list,
        description="Reader names (from readers.yaml) to ask in this round.",
    )
    crop_margins: list[float] = Field(
        default_factory=list,
        description=(
            "Extra views with the region grown by this fraction of its own "
            "width/height on every side (0.1 = 10%). Empty = base crop only."
        ),
    )
    variants: list[str] = Field(
        default_factory=list,
        description="Named image variants (alt preprocessing) to also read under.",
    )

    @model_validator(mode="after")
    def _margins_sane(self) -> EscalationTier:
        if any(m <= 0.0 or m > 1.0 for m in self.crop_margins):
            msg = "crop_margins must be in (0, 1]"
            raise ValueError(msg)
        if not self.readers:
            msg = "an escalation tier needs at least one reader"
            raise ValueError(msg)
        return self


class EscalationConfig(BaseModel):
    """Escalation on disagreement (P6.5). Tiers run in order, cheapest first."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = False
    tiers: list[EscalationTier] = Field(default_factory=list)
    max_extra_reads: int = Field(
        default=24,
        ge=1,
        description="Hard cap on extra reader calls per region across all tiers.",
    )


class RegressionGates(BaseModel):
    """Eval regression thresholds (``torus.eval diff``)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    noise_margin: float = Field(
        ge=0.0,
        le=1.0,
        description="Max allowed absolute accuracy drop per column.",
    )
    silent_error_tolerance: int = Field(
        ge=0,
        description="Max allowed increase in silent errors vs baseline.",
    )


class IngestConfig(BaseModel):
    """PDF/image ingest knobs (embedded-text usability thresholds)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    embedded_text_min_chars: int = Field(
        default=25,
        ge=0,
        description=(
            "Minimum non-whitespace characters before text layer is considered."
        ),
    )
    embedded_text_max_nonprintable_ratio: float = Field(
        default=0.02,
        ge=0.0,
        le=1.0,
        description="Reject text layer if this fraction of chars is non-printable.",
    )
    embedded_text_max_order_violation_ratio: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        description=(
            "Reject text layer if this fraction of consecutive word pairs "
            "violates top-to-bottom / left-to-right reading order."
        ),
    )
    embedded_text_reject_cid_garbage: bool = Field(
        default=True,
        description="Reject text layers that contain PDF (cid:N) extraction garbage.",
    )


class OrientationConfig(BaseModel):
    """Orientation detection knobs (classic + classifier, P2.2)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = Field(
        default=True,
        description=(
            "When false, pages stay as ingested and orientation is reported as 0."
        ),
    )


class DeskewConfig(BaseModel):
    """Small-angle skew correction (P2.3)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = True
    min_angle_deg: float = Field(
        default=0.4,
        ge=0.0,
        le=45.0,
        description="Ignore detected skew below this absolute angle (degrees).",
    )
    max_angle_deg: float = Field(
        default=15.0,
        ge=0.0,
        le=45.0,
        description="Ignore detections above this absolute angle (not a card skew).",
    )

    @model_validator(mode="after")
    def _min_not_above_max(self) -> DeskewConfig:
        if self.min_angle_deg > self.max_angle_deg:
            msg = (
                "min_angle_deg must be <= max_angle_deg, "
                f"got {self.min_angle_deg} > {self.max_angle_deg}"
            )
            raise ValueError(msg)
        return self


class PerspectiveConfig(BaseModel):
    """Photo document-plane correction (P2.3)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = True
    min_quad_area_ratio: float = Field(
        default=0.18,
        ge=0.0,
        le=1.0,
        description="Minimum page-area fraction for an accepted document quad.",
    )
    max_quad_area_ratio: float = Field(
        default=0.96,
        ge=0.0,
        le=1.0,
        description=(
            "Reject quads that already fill nearly the whole frame "
            "(likely a flat scan, not a photo)."
        ),
    )

    @model_validator(mode="after")
    def _min_not_above_max(self) -> PerspectiveConfig:
        if self.min_quad_area_ratio > self.max_quad_area_ratio:
            msg = (
                "min_quad_area_ratio must be <= max_quad_area_ratio, "
                f"got {self.min_quad_area_ratio} > {self.max_quad_area_ratio}"
            )
            raise ValueError(msg)
        return self


class CropConfig(BaseModel):
    """Border and shadow margin cropping (P2.3)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = True
    margin_px: int = Field(
        default=8,
        ge=0,
        description="Pixels of padding kept around detected content.",
    )
    min_content_ratio: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Skip crop when ink/content covers less than this of the page.",
    )
    max_trim_ratio: float = Field(
        default=0.35,
        ge=0.0,
        le=0.5,
        description="Never trim more than this fraction from any single side.",
    )


class DenoiseConfig(BaseModel):
    """Non-local means denoise for fax grain (P2.4)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = True
    h: float = Field(
        default=8.0,
        ge=0.0,
        le=30.0,
        description="Filter strength for cv2.fastNlMeansDenoisingColored.",
    )
    template_window_size: int = Field(
        default=7,
        ge=3,
        le=15,
        description="Odd template patch size in pixels.",
    )
    search_window_size: int = Field(
        default=21,
        ge=5,
        le=35,
        description="Odd search window size in pixels.",
    )

    @field_validator("template_window_size", "search_window_size")
    @classmethod
    def _odd_window(cls, value: int) -> int:
        if value % 2 == 0:
            msg = f"window size must be odd, got {value}"
            raise ValueError(msg)
        return value


class ContrastNormConfig(BaseModel):
    """CLAHE contrast normalisation for faded / grey scans (P2.4)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = True
    clip_limit: float = Field(
        default=2.0,
        ge=0.1,
        le=10.0,
        description="CLAHE clip limit; higher amplifies local contrast more.",
    )
    tile_grid_size: int = Field(
        default=8,
        ge=2,
        le=32,
        description="CLAHE tile grid size (tiles per side).",
    )


class StrokeThickenConfig(BaseModel):
    """Morphological stroke thickening for thin fax lines (P2.4)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = True
    kernel_size: int = Field(
        default=2,
        ge=1,
        le=5,
        description="Elliptical dilation kernel size in pixels.",
    )


class QualityRepairConfig(BaseModel):
    """Fax-like / faded page repair; each step is independently switchable."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = Field(
        default=True,
        description="Master switch; when false no repair step runs.",
    )
    # Order is fixed in code: denoise → contrast_norm → stroke_thicken.
    denoise: DenoiseConfig = Field(default_factory=DenoiseConfig)
    contrast_norm: ContrastNormConfig = Field(default_factory=ContrastNormConfig)
    stroke_thicken: StrokeThickenConfig = Field(default_factory=StrokeThickenConfig)


class BlankDetectConfig(BaseModel):
    """Blank / near-blank page detection (ink density + text-presence check)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = True
    max_ink_density: float = Field(
        default=0.004,
        ge=0.0,
        le=1.0,
        description="Page is blank only when ink density is at or below this.",
    )
    min_text_components: int = Field(
        default=3,
        ge=1,
        description=(
            "Classical OCR-presence gate: blank only when glyph-like connected "
            "components are strictly fewer than this count."
        ),
    )
    min_component_area: int = Field(
        default=12,
        ge=1,
        description="Ignore speckles smaller than this many pixels.",
    )
    max_component_area_ratio: float = Field(
        default=0.02,
        ge=0.0,
        le=1.0,
        description="Ignore blobs larger than this fraction of the page area.",
    )
    min_component_height_ratio: float = Field(
        default=0.004,
        ge=0.0,
        le=1.0,
        description="Minimum component height as a fraction of page height.",
    )
    max_component_height_ratio: float = Field(
        default=0.08,
        ge=0.0,
        le=1.0,
        description="Maximum component height as a fraction of page height.",
    )
    max_component_aspect: float = Field(
        default=6.0,
        ge=1.0,
        description="Reject components wider/taller than this aspect ratio.",
    )

    @model_validator(mode="after")
    def _height_ratio_order(self) -> BlankDetectConfig:
        if self.min_component_height_ratio > self.max_component_height_ratio:
            msg = (
                "min_component_height_ratio must be <= max_component_height_ratio, "
                f"got {self.min_component_height_ratio} > "
                f"{self.max_component_height_ratio}"
            )
            raise ValueError(msg)
        return self


class PageQualityConfig(BaseModel):
    """Blur / contrast / resolution scoring for Page.quality_score (P2.6)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = Field(
        default=True,
        description=(
            "When false, quality scoring is skipped and Page fields are left alone."
        ),
    )
    weight_blur: float = Field(
        default=1.0,
        ge=0.0,
        description="Relative weight of the blur component in the combined score.",
    )
    weight_contrast: float = Field(
        default=1.0,
        ge=0.0,
        description="Relative weight of the contrast component.",
    )
    weight_resolution: float = Field(
        default=1.0,
        ge=0.0,
        description="Relative weight of the resolution component.",
    )
    blur_laplacian_ref: float = Field(
        default=40.0,
        gt=0.0,
        description=(
            "Denoised Laplacian variance at which the blur component saturates to 1.0."
        ),
    )
    contrast_ref: float = Field(
        default=18.0,
        gt=0.0,
        description=(
            "Grey-channel std at which the contrast component saturates to 1.0."
        ),
    )
    target_min_side_px: int = Field(
        default=1600,
        ge=32,
        description=(
            "Short-side pixel length at which the resolution component "
            "saturates to 1.0."
        ),
    )
    low_quality_threshold: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Pages with quality_score below this are flagged low_quality.",
    )

    @model_validator(mode="after")
    def _at_least_one_weight(self) -> PageQualityConfig:
        if (
            self.weight_blur <= 0.0
            and self.weight_contrast <= 0.0
            and self.weight_resolution <= 0.0
        ):
            msg = (
                "at least one of weight_blur, weight_contrast, "
                "weight_resolution must be > 0"
            )
            raise ValueError(msg)
        return self


class PreprocessConfig(BaseModel):
    """Page clean-up switches (orientation, deskew, repair, blank, quality)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    orientation: OrientationConfig = Field(default_factory=OrientationConfig)
    deskew: DeskewConfig = Field(default_factory=DeskewConfig)
    perspective: PerspectiveConfig = Field(default_factory=PerspectiveConfig)
    crop: CropConfig = Field(default_factory=CropConfig)
    quality_repair: QualityRepairConfig = Field(default_factory=QualityRepairConfig)
    blank_detect: BlankDetectConfig = Field(default_factory=BlankDetectConfig)
    page_quality: PageQualityConfig = Field(default_factory=PageQualityConfig)


class ClassifyStrategy(StrEnum):
    """Which page-classifier backend(s) to run."""

    VLM = "vlm"
    TRAINED = "trained"
    VOTE = "vote"


class ClassifyVlmConfig(BaseModel):
    """Zero-shot VLM classifier (local model server only)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = True
    model: str | None = Field(
        default=None,
        description="Model id on the local server (e.g. Ollama tag).",
    )
    endpoint_env: str = Field(
        default="TORUS_LLM_URL",
        description="Env var holding the local VLM base URL.",
    )
    path: str = Field(
        default="/api/chat",
        description="HTTP path under the local server base URL.",
    )
    temperature: float = Field(default=0.0, ge=0.0, le=1.0)
    timeout_s: float = Field(default=60.0, gt=0.0)
    min_vram_gb: float = Field(default=4.0, ge=0.0)
    min_ram_gb: float = Field(default=8.0, ge=0.0)


class ClassifyTrainedConfig(BaseModel):
    """Small trained image+text page classifier."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = True
    weights: str | None = Field(
        default=None,
        description=(
            "Weights file path. Relative paths resolve under TORUS_MODEL_DIR. "
            "When missing, a cue+visual prior runs (no crash)."
        ),
    )


class ClassifyVoteConfig(BaseModel):
    """How to combine VLM and trained predictions when strategy is vote."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    agree_boost: float = Field(
        default=0.1,
        ge=0.0,
        le=0.5,
        description="Confidence added when both strategies agree (capped at 1).",
    )
    disagree_penalty: float = Field(
        default=0.15,
        ge=0.0,
        le=0.5,
        description="Confidence subtracted from the winner when they disagree.",
    )


class ClassifyConfig(BaseModel):
    """Page-type classification knobs (Phase 3)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = True
    strategy: ClassifyStrategy = ClassifyStrategy.VOTE
    fallback_page_type: str = Field(
        default="other",
        min_length=1,
        description="Used when no strategy can run or the label is unknown.",
    )
    vlm: ClassifyVlmConfig = Field(default_factory=ClassifyVlmConfig)
    trained: ClassifyTrainedConfig = Field(default_factory=ClassifyTrainedConfig)
    vote: ClassifyVoteConfig = Field(default_factory=ClassifyVoteConfig)


class LayoutClassicConfig(BaseModel):
    """OpenCV classic layout detectors (P4.1). Ratios are relative to page size."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = True
    nms_iou: float = Field(default=0.5, ge=0.0, le=1.0)

    detect_text_lines: bool = True
    detect_text_blocks: bool = True
    detect_tables: bool = True
    detect_stamps: bool = True
    detect_signatures: bool = True
    detect_photos: bool = True

    line_close_width_ratio: float = Field(default=0.02, gt=0.0, le=0.2)
    line_min_area_ratio: float = Field(default=0.00005, ge=0.0, le=0.1)
    line_min_width_ratio: float = Field(default=0.03, ge=0.0, le=1.0)
    line_max_height_ratio: float = Field(default=0.08, gt=0.0, le=0.5)
    line_min_aspect: float = Field(default=1.4, ge=1.0)

    block_min_lines: int = Field(default=2, ge=1)
    block_max_line_gap_ratio: float = Field(default=1.2, ge=0.0, le=5.0)

    table_line_min_length_ratio: float = Field(default=0.03, gt=0.0, le=1.0)
    table_min_cell_area_ratio: float = Field(default=0.0004, ge=0.0, le=0.2)
    table_max_cell_area_ratio: float = Field(default=0.08, gt=0.0, le=0.5)
    table_min_cells: int = Field(default=4, ge=1)

    stamp_min_saturation: int = Field(default=70, ge=0, le=255)
    stamp_min_value: int = Field(default=40, ge=0, le=255)
    stamp_min_radius_ratio: float = Field(default=0.02, gt=0.0, le=0.5)
    stamp_max_radius_ratio: float = Field(default=0.18, gt=0.0, le=0.5)
    stamp_min_area_ratio: float = Field(default=0.0015, ge=0.0, le=0.2)
    stamp_hough_param2: int = Field(default=35, ge=1)

    signature_min_area_ratio: float = Field(default=0.0002, ge=0.0, le=0.1)
    signature_max_area_ratio: float = Field(default=0.05, gt=0.0, le=0.5)
    signature_min_aspect: float = Field(default=1.8, ge=1.0)
    signature_max_height_ratio: float = Field(default=0.15, gt=0.0, le=0.5)
    signature_max_fill_ratio: float = Field(default=0.45, gt=0.0, le=1.0)

    # Absolute per-pixel channel variance (not 0–255 normalised).
    photo_variance_threshold: int = Field(default=80, ge=0)
    photo_text_edge_density: float = Field(default=40.0, ge=0.0)
    photo_min_area_ratio: float = Field(default=0.008, ge=0.0, le=0.5)
    photo_max_area_ratio: float = Field(default=0.35, gt=0.0, le=1.0)
    photo_min_aspect: float = Field(default=0.45, gt=0.0)
    photo_max_aspect: float = Field(default=2.2, gt=0.0)
    photo_max_edge_ratio: float = Field(default=0.22, ge=0.0, le=1.0)


class LayoutModelConfig(BaseModel):
    """Optional neural layout model (disabled until a dependency is approved)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = False
    backend: str | None = Field(
        default=None,
        description=(
            "Import/registry name for a LayoutModelBackend. "
            "None or unknown → skipped (NullLayoutModel)."
        ),
    )
    min_vram_gb: float = Field(default=0.0, ge=0.0)
    min_ram_gb: float = Field(default=0.0, ge=0.0)


class LabelAnchorConfig(BaseModel):
    """Label → field matching and relative value search (P4.2)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = True
    # Cosine similarity floor for embedding matches (char n-grams).
    embedding_min: float = Field(default=0.72, ge=0.0, le=1.0)
    # Minimum vertical overlap ratio (intersection / min height) for same-row / right.
    row_y_overlap_min: float = Field(default=0.35, ge=0.0, le=1.0)
    # Minimum horizontal overlap ratio for "below" candidates.
    below_x_overlap_min: float = Field(default=0.2, ge=0.0, le=1.0)
    # Max gap from label edge to value, as a fraction of page width/height.
    max_right_gap: float = Field(default=0.45, ge=0.0, le=1.0)
    max_below_gap: float = Field(default=0.25, ge=0.0, le=1.0)
    # Search order when several relative placements are possible.
    prefer: list[str] = Field(
        default_factory=lambda: ["right", "same_row", "below"],
        description="Relative placements: right | same_row | below.",
    )
    # Region kinds considered as label candidates and as value candidates.
    label_kinds: list[str] = Field(
        default_factory=lambda: ["text_line", "table_cell"],
    )
    value_kinds: list[str] = Field(
        default_factory=lambda: [
            "text_line",
            "field_value",
            "comb",
            "table_cell",
        ],
    )

    @field_validator("prefer", "label_kinds", "value_kinds")
    @classmethod
    def _non_empty_entries(cls, value: list[str]) -> list[str]:
        for item in value:
            if not item:
                msg = "entries must be non-empty strings"
                raise ValueError(msg)
        return value

    @model_validator(mode="after")
    def _prefer_known(self) -> LabelAnchorConfig:
        allowed = {"right", "same_row", "below"}
        unknown = [p for p in self.prefer if p not in allowed]
        if unknown:
            msg = f"prefer entries must be one of {sorted(allowed)}, got {unknown}"
            raise ValueError(msg)
        return self


class LayoutCombConfig(BaseModel):
    """Comb / character-box detection, line inpaint, cell split (P4.3)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = True
    min_cells: int = Field(default=3, ge=2)
    max_cells: int = Field(default=32, ge=2)
    # Outer comb aspect (width / height); character rows are wide.
    min_aspect: float = Field(default=1.6, ge=1.0)
    # Max relative std-dev of consecutive cell widths (equal-pitch grid).
    max_pitch_cv: float = Field(default=0.22, ge=0.0, le=1.0)
    # Vertical divider kernel length as a fraction of candidate height.
    v_line_height_ratio: float = Field(default=0.55, gt=0.0, le=1.0)
    # Horizontal ruling kernel length as a fraction of candidate width.
    h_line_width_ratio: float = Field(default=0.35, gt=0.0, le=1.0)
    # Minimum fraction of ROI height a vertical ruling must cover.
    v_cover_min: float = Field(default=0.45, ge=0.0, le=1.0)
    # Ink threshold relative to local mean (lower → more ink kept).
    ink_threshold: int = Field(default=140, ge=0, le=255)
    # Half-width (px) used when protecting strokes that cross dividers.
    stroke_protect_radius: int = Field(default=2, ge=1, le=8)
    # cv2.inpaint radius for removing ruled lines after stroke protection.
    inpaint_radius: int = Field(default=2, ge=1, le=8)
    # Prefer COMB over overlapping TABLE_CELL when IoU exceeds this.
    prefer_over_table_iou: float = Field(default=0.25, ge=0.0, le=1.0)


class LayoutCheckboxConfig(BaseModel):
    """Checkbox / radio detection, fill-state, label mapping (P4.4)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = True
    # Control size as a fraction of min(page width, page height).
    min_side_ratio: float = Field(default=0.015, gt=0.0, le=0.2)
    max_side_ratio: float = Field(default=0.35, gt=0.0, le=0.5)
    min_aspect: float = Field(default=0.7, gt=0.0)
    max_aspect: float = Field(default=1.4, gt=0.0)
    # Reject solid blobs (photos) whose interior fill exceeds this.
    max_solid_fill: float = Field(default=0.72, ge=0.0, le=1.0)
    # Contour circularity floor for radio fallback (4πA/P²).
    min_circularity: float = Field(default=0.65, ge=0.0, le=1.0)
    hough_param2: int = Field(default=22, ge=1)
    ink_threshold: int = Field(default=140, ge=0, le=255)
    nms_iou: float = Field(default=0.4, ge=0.0, le=1.0)
    # Fill-state thresholds (fraction of interior dark pixels).
    empty_fill_max: float = Field(default=0.08, ge=0.0, le=1.0)
    tick_fill_min: float = Field(default=0.08, ge=0.0, le=1.0)
    cross_fill_min: float = Field(default=0.12, ge=0.0, le=1.0)
    diagonal_ink_min: float = Field(default=0.22, ge=0.0, le=1.0)
    diagonal_asymmetry_min: float = Field(default=0.12, ge=0.0, le=1.0)
    # Adjacent option-label search (normalised page units).
    max_label_gap: float = Field(default=0.35, ge=0.0, le=1.0)
    row_y_overlap_min: float = Field(default=0.25, ge=0.0, le=1.0)


class LayoutTableConfig(BaseModel):
    """Table cell / header / row structure and cross-page stitch (P4.5)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = True
    # How many leading rows are treated as headers (geometry-only; no OCR).
    header_rows: int = Field(default=1, ge=0, le=8)
    # Minimum cells and columns before a cluster counts as a table.
    min_cells: int = Field(default=4, ge=1)
    min_columns: int = Field(default=2, ge=1)
    min_rows: int = Field(default=2, ge=1)
    # Row / column clustering: centres within this multiple of median cell size.
    row_cluster_tol: float = Field(default=0.55, gt=0.0, le=2.0)
    col_cluster_tol: float = Field(default=0.55, gt=0.0, le=2.0)
    # Split one page's cells into separate tables when gap exceeds this
    # multiple of median cell height (normalised).
    table_gap_tol: float = Field(default=1.8, gt=0.0, le=10.0)
    nms_iou: float = Field(default=0.4, ge=0.0, le=1.0)
    # Cross-page continuation: max |Δx| of matching column centres (norm).
    col_align_tol: float = Field(default=0.04, ge=0.0, le=0.5)
    # First page table must extend into the bottom band; continuation into top.
    page_edge_band: float = Field(default=0.22, gt=0.0, le=0.5)


class LayoutStampConfig(BaseModel):
    """Stamp / signature detection, overlay mask, signed yes/no (P4.6)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = True
    detect_stamps: bool = True
    detect_signatures: bool = True
    nms_iou: float = Field(default=0.4, ge=0.0, le=1.0)
    # HSV chroma thresholds for stamp ink (red / blue seals).
    stamp_min_saturation: int = Field(default=55, ge=0, le=255)
    stamp_min_value: int = Field(default=40, ge=0, le=255)
    # Signature geometry (ratios of page size); slightly looser than classic
    # defaults so sparse strokes over forms still register for signed yes/no.
    signature_min_area_ratio: float = Field(default=0.00005, ge=0.0, le=0.1)
    signature_max_area_ratio: float = Field(default=0.08, gt=0.0, le=0.5)
    signature_min_aspect: float = Field(default=1.6, ge=1.0)
    signature_max_height_ratio: float = Field(default=0.2, gt=0.0, le=0.5)
    signature_max_fill_ratio: float = Field(default=0.55, gt=0.0, le=1.0)
    # Dark-ink floor for signature stroke masks (0–255 gray).
    signature_ink_threshold: int = Field(default=140, ge=0, le=255)
    # Gray darker than this is treated as printed text and kept out of stamp mask.
    protect_text_max_gray: int = Field(default=70, ge=0, le=255)
    # Dilate the overlay mask slightly so translucent edges are covered.
    mask_dilate_px: int = Field(default=2, ge=0, le=8)
    # cv2.inpaint radius after building the overlay mask.
    inpaint_radius: int = Field(default=3, ge=1, le=12)


class FormFamilyStatus(StrEnum):
    """Lifecycle of an auto-learned form-family hint file."""

    OBSERVING = "observing"
    GENERATED = "generated"
    REVIEWED = "reviewed"
    REJECTED = "rejected"


class FamilyPoint(BaseModel):
    """Normalised page point (0–1). Lives in config only — never hard-coded in code."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)


class FamilySize(BaseModel):
    """Normalised width/height of a label box."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    w: float = Field(gt=0.0, le=1.0)
    h: float = Field(gt=0.0, le=1.0)


class FamilyOffset(BaseModel):
    """Signed centre offset from label centre to value centre (page fraction)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    dx: float = Field(ge=-1.0, le=1.0)
    dy: float = Field(ge=-1.0, le=1.0)


class FamilyLabelHint(BaseModel):
    """One field's label location and preferred value placement (hint only)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    field_key: str = Field(min_length=1)
    label_center: FamilyPoint
    label_size: FamilySize
    preferred_placement: str = Field(
        default="right",
        description="Relative placement: right | same_row | below.",
    )
    value_offset: FamilyOffset | None = Field(
        default=None,
        description="Optional centre offset from label centre to value centre.",
    )

    @field_validator("preferred_placement")
    @classmethod
    def _known_placement(cls, value: str) -> str:
        allowed = {"right", "same_row", "below"}
        if value not in allowed:
            msg = f"preferred_placement must be one of {sorted(allowed)}, got {value!r}"
            raise ValueError(msg)
        return value


class FormFamilySpec(BaseModel):
    """Auto-learned printed-form label layout under ``config/families/``.

    Hints only: the pipeline must locate fields without any family file.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1)
    status: FormFamilyStatus = FormFamilyStatus.OBSERVING
    fingerprint: str = Field(min_length=1)
    observation_count: int = Field(default=1, ge=1)
    template_id: str | None = None
    page_type: str | None = None
    labels: list[FamilyLabelHint] = Field(default_factory=list)
    notes: str | None = Field(
        default=None,
        description="Human review notes; never document content.",
    )

    @model_validator(mode="after")
    def _unique_field_keys(self) -> FormFamilySpec:
        keys = [lab.field_key for lab in self.labels]
        if len(keys) != len(set(keys)):
            msg = "family label field_key values must be unique"
            raise ValueError(msg)
        return self


class LayoutFamiliesConfig(BaseModel):
    """Form-family auto-save and hint use (P4.7)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = True
    # Write/update YAML under config/families/ when a layout is seen this often.
    min_observations: int = Field(default=2, ge=1)
    # Minimum anchored labels before a layout can form a family.
    min_labels: int = Field(default=2, ge=2)
    # Quantisation bin for translation-invariant fingerprints (page fraction).
    position_bin: float = Field(default=0.05, gt=0.0, le=0.5)
    # Soft match: max centre distance (norm) when fingerprint bins disagree.
    match_tolerance: float = Field(default=0.08, gt=0.0, le=0.5)
    # Statuses that may bias anchoring (hints never required for correctness).
    use_as_hint: list[str] = Field(
        default_factory=lambda: ["generated", "reviewed"],
    )
    auto_save: bool = True

    @field_validator("use_as_hint")
    @classmethod
    def _known_statuses(cls, value: list[str]) -> list[str]:
        allowed = {s.value for s in FormFamilyStatus}
        for item in value:
            if item not in allowed:
                msg = (
                    f"use_as_hint entry must be one of {sorted(allowed)}, got {item!r}"
                )
                raise ValueError(msg)
        return value


class LayoutConfig(BaseModel):
    """Layout / field-location knobs (Phase 4)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = True
    classic: LayoutClassicConfig = Field(default_factory=LayoutClassicConfig)
    model: LayoutModelConfig = Field(default_factory=LayoutModelConfig)
    anchor: LabelAnchorConfig = Field(default_factory=LabelAnchorConfig)
    comb: LayoutCombConfig = Field(default_factory=LayoutCombConfig)
    checkbox: LayoutCheckboxConfig = Field(default_factory=LayoutCheckboxConfig)
    table: LayoutTableConfig = Field(default_factory=LayoutTableConfig)
    stamp: LayoutStampConfig = Field(default_factory=LayoutStampConfig)
    families: LayoutFamiliesConfig = Field(default_factory=LayoutFamiliesConfig)
    # Location match threshold used by layout eval (P4.8) and component tests.
    locate_iou: float = Field(default=0.3, ge=0.0, le=1.0)


class MappingConfig(BaseModel):
    """Local LLM field mapper (P8.3). Talks only to a local/internal server."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = False
    model: str | None = Field(
        default=None,
        description="Model id on the local server (a Qwen3 size, e.g. an Ollama tag).",
    )
    endpoint_env: str = Field(
        default="TORUS_LLM_URL",
        description="Env var holding the local LLM base URL.",
    )
    path: str = Field(
        default="/api/chat",
        description="HTTP path; /api/chat (Ollama) or an OpenAI-style /v1 path.",
    )
    temperature: float = Field(default=0.0, ge=0.0, le=0.0)
    seed: int = 0
    timeout_s: float = Field(default=120.0, gt=0.0)
    min_vram_gb: float = Field(default=6.0, ge=0.0)
    min_ram_gb: float = Field(default=8.0, ge=0.0)


class ReviewConfig(BaseModel):
    """Review metrics policy (P9.5)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_gap_s: float = Field(
        default=120.0,
        gt=0.0,
        description=(
            "Pause between two decisions above which the gap is a break, not "
            "review time, and is left out of seconds per field."
        ),
    )


class OutputConfig(BaseModel):
    """Excel writer policy (P8.5). Presentation and safety, never extraction."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    overwrite_existing: bool = Field(
        default=False,
        description=(
            "When false, a template cell that already holds anything is protected "
            "and left untouched. Passthrough columns are never written either way."
        ),
    )
    write_statuses: list[FieldStatus] = Field(
        default_factory=lambda: [
            FieldStatus.CONFIRMED,
            FieldStatus.LIKELY,
            FieldStatus.NEEDS_REVIEW,
        ],
        description=(
            "Statuses whose value is written into the template cell. Others stay "
            "empty (and are still listed on the evidence sheet)."
        ),
    )
    data_start_row: int | None = Field(
        default=None,
        ge=1,
        description="First output row (1-based). Null = the row below the header.",
    )
    evidence_sheet: str = Field(
        default="torus_evidence",
        min_length=1,
        max_length=31,
        description="Name of the added status/confidence/evidence sheet.",
    )
    colors: dict[FieldStatus, str] = Field(
        default_factory=lambda: {
            FieldStatus.CONFIRMED: "C6EFCE",
            FieldStatus.LIKELY: "FFEB9C",
            FieldStatus.NEEDS_REVIEW: "FFC7CE",
            FieldStatus.NOT_FOUND: "D9D9D9",
        },
        description="Fill colour (RRGGBB) per status.",
    )
    reviewed_color: str = Field(
        default="BDD7EE",
        description="Fill colour (RRGGBB) for cells a human reviewed (P9.3).",
    )

    @field_validator("reviewed_color")
    @classmethod
    def _reviewed_color_hex(cls, value: str) -> str:
        if len(value) != 6 or any(c not in "0123456789abcdefABCDEF" for c in value):
            msg = f"reviewed_color must be RRGGBB hex, got {value!r}"
            raise ValueError(msg)
        return value

    @field_validator("colors")
    @classmethod
    def _colors_complete_hex(
        cls, value: dict[FieldStatus, str]
    ) -> dict[FieldStatus, str]:
        missing = [s.value for s in FieldStatus if s not in value]
        if missing:
            msg = "colors must cover every status; missing: " + ", ".join(missing)
            raise ValueError(msg)
        for status, code in value.items():
            if len(code) != 6 or any(c not in "0123456789abcdefABCDEF" for c in code):
                msg = f"colors[{status.value}] must be RRGGBB hex, got {code!r}"
                raise ValueError(msg)
        return value


class PipelineConfig(BaseModel):
    """Thresholds and knobs shared across the pipeline."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    dpi: int = Field(ge=72, le=1200)
    ingest: IngestConfig = Field(default_factory=IngestConfig)
    preprocess: PreprocessConfig = Field(default_factory=PreprocessConfig)
    classify: ClassifyConfig = Field(default_factory=ClassifyConfig)
    layout: LayoutConfig = Field(default_factory=LayoutConfig)
    mapping: MappingConfig = Field(default_factory=MappingConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    review: ReviewConfig = Field(default_factory=ReviewConfig)
    confidence: ConfidenceThresholds
    voting: VotingRules
    escalation: EscalationConfig = Field(default_factory=EscalationConfig)
    regression: RegressionGates


class ReaderSpec(BaseModel):
    """One plug-in reader entry from ``config/readers.yaml``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    family: str = Field(min_length=1)
    enabled: bool = False
    min_vram_gb: float = Field(ge=0.0)
    min_ram_gb: float = Field(ge=0.0)
    device: DeviceKind = DeviceKind.AUTO
    capabilities: ReaderCaps = Field(default_factory=ReaderCaps)
    model: str | None = None
    version: str | None = None
    registry_version: str | None = Field(
        default=None,
        description=(
            "Model-registry version (P10.3): a version id or 'current'. "
            "torus.learn.registry.apply_registry replaces model/version with it."
        ),
    )
    module: str | None = Field(
        default=None,
        description="Import path of the reader class (wired in Phase 5).",
    )


class ReadersConfig(BaseModel):
    """Registry of readers. Disabled or missing readers are skipped, never a crash."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    # Runtime knobs for the reader framework (P5.1). Defaults keep older YAML valid.
    timeout_s: float = Field(
        default=30.0,
        gt=0.0,
        description="Per-read wall-clock timeout; soft-fail as Reading.error.",
    )
    batch_size: int = Field(
        default=8,
        ge=1,
        description="Max regions per batch when a reader exposes read_batch.",
    )
    cache_max_entries: int = Field(
        default=1024,
        ge=0,
        description="LRU cache size (crop hash + reader version). 0 disables.",
    )
    readers: list[ReaderSpec] = Field(default_factory=list)

    @model_validator(mode="after")
    def _unique_names(self) -> ReadersConfig:
        names = [r.name for r in self.readers]
        if len(names) != len(set(names)):
            msg = "reader names must be unique"
            raise ValueError(msg)
        return self

    def get(self, name: str) -> ReaderSpec | None:
        for reader in self.readers:
            if reader.name == name:
                return reader
        return None


class PageTypeSpec(BaseModel):
    """One page type the classifier may assign."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    examples: list[str] = Field(default_factory=list)
    visual_cues: list[str] = Field(default_factory=list)
    text_cues: list[str] = Field(default_factory=list)


class BlankPolicy(StrEnum):
    """How blank pages participate in pack splitting (P3.3)."""

    ATTACH_PREVIOUS = "attach_previous"
    OWN_GROUP = "own_group"
    SKIP = "skip"


class PackSplitConfig(BaseModel):
    """Rules for grouping consecutive pages into logical documents.

    Document knowledge stays here — no per-sample page indices in code.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    merge_same: list[str] = Field(
        default_factory=list,
        description=(
            "Page-type ids that merge when consecutive pages share that type "
            "(e.g. multi-page statements)."
        ),
    )
    continuations: dict[str, list[str]] = Field(
        default_factory=dict,
        description=(
            "Map of continuation type id → starter/prior types it may follow "
            "inside the same logical document."
        ),
    )
    blank_policy: BlankPolicy = BlankPolicy.ATTACH_PREVIOUS
    blank_page_type: str = Field(
        default="blank",
        min_length=1,
        description="Page-type id treated as blank for splitting rules.",
    )

    @field_validator("merge_same")
    @classmethod
    def _non_empty_merge_same(cls, value: list[str]) -> list[str]:
        cleaned: list[str] = []
        seen: set[str] = set()
        for item in value:
            stripped = item.strip()
            if not stripped:
                msg = "merge_same entries must be non-empty strings"
                raise ValueError(msg)
            if stripped not in seen:
                cleaned.append(stripped)
                seen.add(stripped)
        return cleaned

    @field_validator("continuations")
    @classmethod
    def _non_empty_continuations(
        cls, value: dict[str, list[str]]
    ) -> dict[str, list[str]]:
        cleaned: dict[str, list[str]] = {}
        for key, priors in value.items():
            k = key.strip()
            if not k:
                msg = "continuations keys must be non-empty"
                raise ValueError(msg)
            prior_clean: list[str] = []
            seen: set[str] = set()
            for prior in priors:
                p = prior.strip()
                if not p:
                    msg = f"continuations[{k!r}] entries must be non-empty"
                    raise ValueError(msg)
                if p not in seen:
                    prior_clean.append(p)
                    seen.add(p)
            if not prior_clean:
                msg = f"continuations[{k!r}] must list at least one prior type"
                raise ValueError(msg)
            cleaned[k] = prior_clean
        return cleaned


class PersonRoleSpec(BaseModel):
    """One person role (borrower / co-applicant / …) — role labels only, never names."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1)
    text_cues: list[str] = Field(
        default_factory=list,
        description="Role phrases matched in page text (structural cues, not names).",
    )


class MultiPersonConfig(BaseModel):
    """Rules for assigning pages to person roles in multi-person packs (P3.4)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    roles: list[PersonRoleSpec] = Field(default_factory=list)
    default_role_order: list[str] = Field(
        default_factory=list,
        description="Role ids used in order when no text cue is found.",
    )
    overflow_role: str = Field(
        default="other_person",
        min_length=1,
        description="Role id when more people appear than default_role_order.",
    )
    person_bearing_types: list[str] = Field(
        default_factory=list,
        description="Page-type ids that belong to a person (forms, IDs, statements).",
    )
    shared_types: list[str] = Field(
        default_factory=list,
        description="Page-type ids that are pack-level (no person role).",
    )
    person_boundary_types: list[str] = Field(
        default_factory=list,
        description=(
            "Page-type ids that start a new person when they appear after an "
            "existing person already has content (e.g. a second application_form)."
        ),
    )
    blank_page_type: str = Field(
        default="blank",
        min_length=1,
        description="Page-type id treated as blank / non-person.",
    )

    @field_validator(
        "default_role_order",
        "person_bearing_types",
        "shared_types",
        "person_boundary_types",
    )
    @classmethod
    def _non_empty_id_lists(cls, value: list[str]) -> list[str]:
        cleaned: list[str] = []
        seen: set[str] = set()
        for item in value:
            stripped = item.strip()
            if not stripped:
                msg = "multi_person id lists must not contain empty strings"
                raise ValueError(msg)
            if stripped not in seen:
                cleaned.append(stripped)
                seen.add(stripped)
        return cleaned

    @model_validator(mode="after")
    def _roles_and_order_consistent(self) -> MultiPersonConfig:
        role_ids = [r.id for r in self.roles]
        if len(role_ids) != len(set(role_ids)):
            msg = "multi_person.roles ids must be unique"
            raise ValueError(msg)
        known = frozenset(role_ids)
        if known:
            unknown_order = [r for r in self.default_role_order if r not in known]
            if unknown_order:
                msg = (
                    "multi_person.default_role_order references unknown roles: "
                    + ", ".join(unknown_order)
                )
                raise ValueError(msg)
        overlap = set(self.person_bearing_types) & set(self.shared_types)
        if overlap:
            msg = (
                "multi_person types cannot be both person_bearing and shared: "
                + ", ".join(sorted(overlap))
            )
            raise ValueError(msg)
        unknown_boundary = [
            t for t in self.person_boundary_types if t not in self.person_bearing_types
        ]
        if unknown_boundary:
            msg = (
                "multi_person.person_boundary_types must also be listed in "
                "person_bearing_types: " + ", ".join(unknown_boundary)
            )
            raise ValueError(msg)
        return self


class PageTypesConfig(BaseModel):
    """Catalogue of page types from ``config/page_types.yaml``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    page_types: list[PageTypeSpec] = Field(default_factory=list)
    pack_split: PackSplitConfig = Field(default_factory=PackSplitConfig)
    multi_person: MultiPersonConfig = Field(default_factory=MultiPersonConfig)

    @model_validator(mode="after")
    def _unique_ids_and_pack_split_refs(self) -> PageTypesConfig:
        ids = [p.id for p in self.page_types]
        if len(ids) != len(set(ids)):
            msg = "page type ids must be unique"
            raise ValueError(msg)
        known = frozenset(ids)
        if not known:
            return self
        unknown_merge = [t for t in self.pack_split.merge_same if t not in known]
        if unknown_merge:
            msg = "pack_split.merge_same references unknown page types: " + ", ".join(
                unknown_merge
            )
            raise ValueError(msg)
        for cont_id, priors in self.pack_split.continuations.items():
            if cont_id not in known:
                msg = f"pack_split.continuations key unknown page type: {cont_id}"
                raise ValueError(msg)
            unknown_priors = [t for t in priors if t not in known]
            if unknown_priors:
                msg = (
                    f"pack_split.continuations[{cont_id!r}] references unknown "
                    f"page types: {', '.join(unknown_priors)}"
                )
                raise ValueError(msg)
        mp = self.multi_person
        for label, type_ids in (
            ("person_bearing_types", mp.person_bearing_types),
            ("shared_types", mp.shared_types),
            ("person_boundary_types", mp.person_boundary_types),
        ):
            unknown = [t for t in type_ids if t not in known]
            if unknown:
                msg = (
                    f"multi_person.{label} references unknown page types: "
                    + ", ".join(unknown)
                )
                raise ValueError(msg)
        if mp.blank_page_type not in known and (
            mp.roles or mp.person_bearing_types or mp.shared_types
        ):
            msg = (
                "multi_person.blank_page_type references unknown page type: "
                + mp.blank_page_type
            )
            raise ValueError(msg)
        return self

    def ids(self) -> frozenset[str]:
        return frozenset(p.id for p in self.page_types)

    def get(self, page_type_id: str) -> PageTypeSpec | None:
        for page_type in self.page_types:
            if page_type.id == page_type_id:
                return page_type
        return None


class NormalizeKind(StrEnum):
    """How a row-key part or field value is canonicalised before compare/align."""

    NONE = "none"
    TEXT = "text"
    PERSON = "person"
    ADDRESS_TYPE = "address_type"
    ADDRESS_SOURCE = "address_source"
    DIGITS = "digits"
    ALNUM = "alnum"
    DATE = "date"


class MatchRule(StrEnum):
    """Per-column equality rule used by ``torus.eval`` scoring."""

    EXACT = "exact"
    NORMALISED = "normalised"
    FUZZY = "fuzzy"


class RowKeyPart(BaseModel):
    """One component of the row alignment key (see schemas/deb/row_key.md)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    sources: list[str] = Field(
        min_length=1,
        description="Column keys; first non-empty value wins.",
    )
    normalize: NormalizeKind = NormalizeKind.TEXT
    aliases: dict[str, list[str]] = Field(
        default_factory=dict,
        description=(
            "Canonical form → accepted surface forms "
            "(used with normalize=address_source)."
        ),
    )

    @field_validator("sources")
    @classmethod
    def _non_empty_sources(cls, value: list[str]) -> list[str]:
        cleaned: list[str] = []
        for item in value:
            stripped = item.strip()
            if not stripped:
                msg = "sources entries must be non-empty strings"
                raise ValueError(msg)
            cleaned.append(stripped)
        return cleaned


class RowKeyConfig(BaseModel):
    """Configurable row alignment for evaluation (P1.2).

    Primary parts form the match key. When several gold rows share a primary
    key, the optional disambiguator (e.g. address_line1) breaks ties.
    When omitted on a template, scoring aligns rows by index.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    primary: list[RowKeyPart] = Field(min_length=1)
    disambiguator: RowKeyPart | None = None
    fuzzy_threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Token-overlap threshold for match=fuzzy.",
    )


class DateOrder(StrEnum):
    """Reading order of an ambiguous all-numeric date such as 03/05/1990."""

    DMY = "dmy"
    MDY = "mdy"


class NameCase(StrEnum):
    """Letter-case applied to a name column on output."""

    PRESERVE = "preserve"
    UPPER = "upper"
    LOWER = "lower"
    TITLE = "title"


class OutputFormat(BaseModel):
    """How a column's value is written to the template (P7.5).

    Presentation only: normalisation never changes what the value says. Anything
    that cannot be re-formatted without guessing is left as read and flagged.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    date_format: str = Field(
        default="%Y-%m-%d",
        description="strftime pattern for date columns.",
    )
    date_order: DateOrder = Field(
        default=DateOrder.DMY,
        description="Day/month order of all-numeric input dates.",
    )
    two_digit_year_pivot: int | None = Field(
        default=None,
        ge=0,
        le=99,
        description=(
            "Two-digit years below this map to 20xx, the rest to 19xx. "
            "When null a two-digit year is not guessed and the value is flagged."
        ),
    )
    name_case: NameCase = NameCase.PRESERVE

    @field_validator("date_format")
    @classmethod
    def _date_format_has_parts(cls, value: str) -> str:
        has_day = "%d" in value
        has_month = any(d in value for d in ("%m", "%b", "%B"))
        has_year = any(d in value for d in ("%Y", "%y"))
        if not (has_day and has_month and has_year):
            msg = "date_format must contain day, month and year directives"
            raise ValueError(msg)
        return value


class AddressSplit(BaseModel):
    """Address line columns a long address is re-flowed across (P7.5)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    keys: list[str] = Field(
        min_length=2,
        description="Line column keys in order (first line first).",
    )
    max_chars: int = Field(ge=1, description="Maximum characters per line.")

    @field_validator("keys")
    @classmethod
    def _unique_non_empty(cls, value: list[str]) -> list[str]:
        if any(not k for k in value) or len(value) != len(set(value)):
            msg = "keys must be unique non-empty strings"
            raise ValueError(msg)
        return value


class FieldSpec(BaseModel):
    """One Excel column the pipeline must fill (S1)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    key: str = Field(min_length=1)
    excel_header: str = Field(min_length=1)
    description: str = Field(min_length=1)
    type: FieldType
    charset: Charset = Charset.ANY
    length: int | None = Field(default=None, ge=0)
    min_length: int | None = Field(default=None, ge=0)
    max_length: int | None = Field(default=None, ge=0)
    validators: list[str] = Field(default_factory=list)
    look_in: list[str] = Field(default_factory=list)
    cardinality: Cardinality = Cardinality.PER_PERSON
    critical: bool = False
    match: MatchRule = MatchRule.EXACT
    normalize: NormalizeKind | None = Field(
        default=None,
        description=(
            "Optional override for match=normalised / fuzzy canonicalisation. "
            "When null, derived from field type."
        ),
    )
    synonyms: list[str] = Field(
        default_factory=list,
        description=(
            "Alternate printed label texts (any language/script) used by "
            "label anchoring (P4.2). Matched after normalisation; embeddings "
            "cover near-misses."
        ),
    )
    output: OutputFormat = Field(
        default_factory=OutputFormat,
        description="Output presentation (date format, name case) for P7.5.",
    )
    passthrough: bool = Field(
        default=False,
        description="True for ops/system columns the engine must not fill (P8.1).",
    )
    duplicate_of: str | None = Field(
        default=None,
        description=(
            "If set, this column mirrors another field key (duplicate Excel header)."
        ),
    )
    excel_col: int | None = Field(
        default=None,
        ge=1,
        description="1-based column index in the client Excel sheet, when known.",
    )

    @field_validator("look_in", "validators", "synonyms")
    @classmethod
    def _non_empty_strings(cls, value: list[str]) -> list[str]:
        for item in value:
            if not item:
                msg = "entries must be non-empty strings"
                raise ValueError(msg)
        return value

    @model_validator(mode="after")
    def _length_bounds(self) -> FieldSpec:
        if (
            self.min_length is not None
            and self.max_length is not None
            and self.min_length > self.max_length
        ):
            msg = (
                "min_length must be <= max_length, "
                f"got {self.min_length} > {self.max_length}"
            )
            raise ValueError(msg)
        if self.length is not None:
            if self.min_length is not None and self.length < self.min_length:
                msg = "length must be >= min_length"
                raise ValueError(msg)
            if self.max_length is not None and self.length > self.max_length:
                msg = "length must be <= max_length"
                raise ValueError(msg)
        return self


class TemplateConfig(BaseModel):
    """Client Excel template: the only columns the engine fills.

    Satisfies ``torus.core.protocols.TemplateSpec`` via the ``id`` property.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1)
    version: int = Field(default=1, ge=1)
    description: str = ""
    fields: list[FieldSpec] = Field(default_factory=list)
    row_key: RowKeyConfig | None = Field(
        default=None,
        description="Eval row alignment; when null, rows align by index.",
    )
    address_split: list[AddressSplit] = Field(
        default_factory=list,
        description="Address line groups a long address is re-flowed across.",
    )

    @model_validator(mode="after")
    def _unique_field_keys(self) -> TemplateConfig:
        keys = [f.key for f in self.fields]
        if len(keys) != len(set(keys)):
            msg = f"template {self.id!r}: field keys must be unique"
            raise ValueError(msg)
        key_set = set(keys)
        for group in self.address_split:
            unknown = [k for k in group.keys if k not in key_set]
            if unknown:
                msg = (
                    f"template {self.id!r} address_split references unknown "
                    f"field keys: {', '.join(unknown)}"
                )
                raise ValueError(msg)
        for spec in self.fields:
            if spec.duplicate_of is None:
                continue
            if spec.duplicate_of == spec.key:
                msg = (
                    f"template {self.id!r} field {spec.key!r}: "
                    "duplicate_of cannot reference itself"
                )
                raise ValueError(msg)
            if spec.duplicate_of not in key_set:
                msg = (
                    f"template {self.id!r} field {spec.key!r}: "
                    f"duplicate_of {spec.duplicate_of!r} is not a field key"
                )
                raise ValueError(msg)
        if self.row_key is not None:
            self._validate_row_key_sources(key_set)
        return self

    def _validate_row_key_sources(self, key_set: set[str]) -> None:
        assert self.row_key is not None
        parts = list(self.row_key.primary)
        if self.row_key.disambiguator is not None:
            parts.append(self.row_key.disambiguator)
        for part in parts:
            unknown = [s for s in part.sources if s not in key_set]
            if unknown:
                msg = (
                    f"template {self.id!r} row_key references unknown "
                    f"field keys: {', '.join(unknown)}"
                )
                raise ValueError(msg)

    def field(self, key: str) -> FieldSpec | None:
        for spec in self.fields:
            if spec.key == key:
                return spec
        return None

    def scored_fields(self) -> list[FieldSpec]:
        """Fields the engine fills (excludes passthrough ops/system columns)."""
        return [f for f in self.fields if not f.passthrough]


class LocalOverrides(BaseModel):
    """Optional machine-local overrides (``config/local.yaml``, git-ignored)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    readers: dict[str, bool] = Field(
        default_factory=dict,
        description="Map of reader name → enabled flag.",
    )


class TorusConfig(BaseModel):
    """Fully loaded, validated configuration used by every stage."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    pipeline: PipelineConfig
    readers: ReadersConfig
    page_types: PageTypesConfig
    templates: dict[str, TemplateConfig]
    config_hash: str = Field(min_length=1)

    def template(self, template_id: str) -> TemplateConfig:
        try:
            return self.templates[template_id]
        except KeyError as exc:
            known = ", ".join(sorted(self.templates)) or "(none)"
            msg = f"unknown template {template_id!r}; known: {known}"
            raise KeyError(msg) from exc
