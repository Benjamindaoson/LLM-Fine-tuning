from __future__ import annotations

import json
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Sequence, Tuple

from torch.utils.data import Dataset

from .formatting import encode_response_only, normalize_messages


class IndexedJsonlDataset(Dataset):
    """Lazy JSONL dataset indexed by byte offsets.

    Only offsets are kept in RAM. File descriptors are opened lazily per worker/process
    and reused across samples, which avoids hundreds of thousands of open/close calls
    during multi-epoch training.
    """

    def __init__(
        self,
        paths: Sequence[str],
        tokenizer: Any,
        max_length: int = 4096,
        truncation: str = "keep_end",
        enable_thinking: bool = False,
    ) -> None:
        self.paths = [str(Path(p)) for p in paths]
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.truncation = truncation
        self.enable_thinking = enable_thinking
        self.index: List[Tuple[int, int]] = []
        self.bad_records = 0
        self._handles: Dict[int, BinaryIO] = {}

        for file_idx, path in enumerate(self.paths):
            with open(path, "rb") as f:
                while True:
                    offset = f.tell()
                    line = f.readline()
                    if not line:
                        break
                    if not line.strip():
                        continue
                    try:
                        item = json.loads(line.decode("utf-8"))
                        normalize_messages(item)
                        self.index.append((file_idx, offset))
                    except Exception:
                        self.bad_records += 1
        if not self.index:
            raise ValueError(f"no usable SFT records loaded from {self.paths}")

    def __len__(self) -> int:
        return len(self.index)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_handles"] = {}
        return state

    def _handle(self, file_idx: int) -> BinaryIO:
        handle = self._handles.get(file_idx)
        if handle is None or handle.closed:
            handle = open(self.paths[file_idx], "rb")
            self._handles[file_idx] = handle
        return handle

    def _read(self, idx: int) -> Dict[str, Any]:
        file_idx, offset = self.index[idx]
        f = self._handle(file_idx)
        f.seek(offset)
        return json.loads(f.readline().decode("utf-8"))

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        messages = normalize_messages(self._read(idx))
        return encode_response_only(
            self.tokenizer,
            messages,
            max_length=self.max_length,
            truncation=self.truncation,
            enable_thinking=self.enable_thinking,
        )

    def close(self) -> None:
        for f in self._handles.values():
            try:
                f.close()
            except Exception:
                pass
        self._handles.clear()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
