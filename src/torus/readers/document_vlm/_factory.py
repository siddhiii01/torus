"""Shared factory for named document-VLM adapters over the local-server path."""

from __future__ import annotations

from collections.abc import Callable

from torus.config.schemas import ReaderSpec
from torus.readers.document_vlm.local_client import DocumentVlmClient
from torus.readers.document_vlm.local_server import LocalServerDocumentVlmReader

CreateReader = Callable[..., LocalServerDocumentVlmReader]


def make_create_reader(default_model: str) -> CreateReader:
    """Return a ``create_reader`` bound to ``default_model`` when spec.model is null."""

    def create_reader(
        *,
        device: str,
        spec: ReaderSpec,
        client: DocumentVlmClient | None = None,
    ) -> LocalServerDocumentVlmReader:
        return LocalServerDocumentVlmReader(
            device=device,
            spec=spec,
            client=client,
            model=spec.model or default_model,
        )

    return create_reader
