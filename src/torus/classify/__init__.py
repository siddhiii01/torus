"""Page-type classification and multi-pack splitting.

P3.2 provides two strategies behind ``PageClassifier``:

* ``ZeroShotVlmClassifier`` — zero-shot VLM using ``page_types.yaml``
* ``TrainedImageTextClassifier`` — small image+text model (NumPy)

Config ``pipeline.classify.strategy`` selects ``vlm``, ``trained``, or ``vote``.

P3.3 pack splitting groups consecutive typed pages into ``LogicalDocument``
units using ``page_types.pack_split`` rules.

P3.4 multi-person detection assigns pages to roles (borrower / co-applicant /
guarantor) from config cues and structural order.
"""

from torus.classify.apply import apply_page_type, classify_document, classify_page
from torus.classify.factory import DisabledPageClassifier, build_page_classifier
from torus.classify.pack import LogicalDocument, split_document, split_pack
from torus.classify.persons import (
    LogicalDocumentPersons,
    MultiPersonResult,
    PagePersonAssignment,
    assign_document_persons,
    assign_persons,
    detect_role_cue,
)
from torus.classify.prompt import (
    build_type_catalogue,
    build_zero_shot_messages,
    parse_vlm_response,
)
from torus.classify.trained import (
    LinearWeights,
    TrainedImageTextClassifier,
    load_weights,
    save_weights,
    train_linear_classifier,
)
from torus.classify.vlm import ZeroShotVlmClassifier
from torus.classify.vlm_client import LocalHttpVlmClient, VlmClient
from torus.classify.vote import VotingPageClassifier

__all__ = [
    "DisabledPageClassifier",
    "LinearWeights",
    "LocalHttpVlmClient",
    "LogicalDocument",
    "LogicalDocumentPersons",
    "MultiPersonResult",
    "PagePersonAssignment",
    "TrainedImageTextClassifier",
    "VlmClient",
    "VotingPageClassifier",
    "ZeroShotVlmClassifier",
    "apply_page_type",
    "assign_document_persons",
    "assign_persons",
    "build_page_classifier",
    "build_type_catalogue",
    "build_zero_shot_messages",
    "classify_document",
    "classify_page",
    "detect_role_cue",
    "load_weights",
    "parse_vlm_response",
    "save_weights",
    "split_document",
    "split_pack",
    "train_linear_classifier",
]
