"""Document-VLM family (P5.3): region/page reading via local model servers.

Concrete adapters live as one module per model name. The solid path is
``local_server.LocalServerDocumentVlmReader`` (stdlib HTTP to ``TORUS_LLM_URL``).
In-process stacks stay optional; missing server or model → soft ``Reading.error``.
"""

from torus.readers.document_vlm.local_server import LocalServerDocumentVlmReader

__all__ = ["LocalServerDocumentVlmReader"]
