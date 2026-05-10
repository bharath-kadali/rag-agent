from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, asdict


@dataclass
class DocumentMeta:
    document_id: str
    filename: str
    content_type: str
    created_at: float


class MetadataStore:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        self.path = os.path.join(self.data_dir, "documents.json")

    def _read_all(self) -> list[DocumentMeta]:
        if not os.path.exists(self.path):
            return []
        with open(self.path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return [DocumentMeta(**d) for d in raw]

    def _write_all(self, docs: list[DocumentMeta]) -> None:
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump([asdict(d) for d in docs], f, ensure_ascii=False, indent=2)

    def create(self, filename: str, content_type: str) -> DocumentMeta:
        docs = self._read_all()
        doc = DocumentMeta(
            document_id=str(uuid.uuid4()),
            filename=filename,
            content_type=content_type,
            created_at=time.time(),
        )
        docs.append(doc)
        self._write_all(docs)
        return doc

    def list(self) -> list[DocumentMeta]:
        return sorted(self._read_all(), key=lambda d: d.created_at, reverse=True)

    def get(self, document_id: str) -> DocumentMeta | None:
        for d in self._read_all():
            if d.document_id == document_id:
                return d
        return None

