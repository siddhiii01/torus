"""Apply page-type predictions onto ``Page`` / ``Document`` objects."""

from __future__ import annotations

from torus.core.models import Document, Page, PageTypePrediction
from torus.core.protocols import PageClassifier


def apply_page_type(page: Page, prediction: PageTypePrediction) -> PageTypePrediction:
    """Set ``page.page_type`` from a prediction and return the prediction."""
    page.page_type = prediction.page_type
    return prediction


def classify_page(classifier: PageClassifier, page: Page) -> PageTypePrediction:
    """Classify one page and write ``page.page_type``."""
    return apply_page_type(page, classifier.classify(page))


def classify_document(
    classifier: PageClassifier,
    document: Document,
) -> list[PageTypePrediction]:
    """Classify every page in document order; writes each ``page.page_type``."""
    return [classify_page(classifier, page) for page in document.pages]
