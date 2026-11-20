"""Page clean-up: orientation, deskew, perspective, crop, quality, blank."""

from torus.preprocess.blank import (
    BlankDetectResult,
    apply_blank_detect,
    count_text_like_components,
    detect_blank,
    flag_blank_pages,
)
from torus.preprocess.crop import CropResult, apply_crop, crop_image, detect_content_box
from torus.preprocess.deskew import (
    DeskewResult,
    apply_deskew,
    deskew_image,
    detect_skew,
)
from torus.preprocess.geometry import GeometryResult, correct_geometry
from torus.preprocess.orientation import (
    OrientationResult,
    apply_orientation,
    classic_orientation,
    classifier_orientation,
    detect_orientation,
    upright_image,
)
from torus.preprocess.page_quality import (
    PageQualityResult,
    apply_page_quality,
    measure_blur,
    measure_page_contrast,
    measure_resolution,
    score_components,
)
from torus.preprocess.perspective import (
    PerspectiveResult,
    apply_perspective,
    detect_perspective,
    warp_perspective,
)
from torus.preprocess.quality_repair import (
    QualityRepairResult,
    StepMetrics,
    apply_quality_repair,
    repair_image,
)

__all__ = [
    "BlankDetectResult",
    "CropResult",
    "DeskewResult",
    "GeometryResult",
    "OrientationResult",
    "PageQualityResult",
    "PerspectiveResult",
    "QualityRepairResult",
    "StepMetrics",
    "apply_blank_detect",
    "apply_crop",
    "apply_deskew",
    "apply_orientation",
    "apply_page_quality",
    "apply_perspective",
    "apply_quality_repair",
    "classic_orientation",
    "classifier_orientation",
    "correct_geometry",
    "count_text_like_components",
    "crop_image",
    "deskew_image",
    "detect_blank",
    "detect_content_box",
    "detect_orientation",
    "detect_perspective",
    "detect_skew",
    "flag_blank_pages",
    "measure_blur",
    "measure_page_contrast",
    "measure_resolution",
    "repair_image",
    "score_components",
    "upright_image",
    "warp_perspective",
]
