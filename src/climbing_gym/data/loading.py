"""Shared dataset loading utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any




def load_dataset_source(
    dataset_name: str,
    dataset_config: str,
    dataset_path: str,
):
    from datasets import load_dataset, load_from_disk

    if dataset_path:
        path = Path(dataset_path)
        if path.is_file():
            suffix = path.suffix.lower()
            if suffix in {".json", ".jsonl"}:
                return load_dataset("json", data_files=str(path))
            if suffix == ".csv":
                return load_dataset("csv", data_files=str(path))
            if suffix in {".parquet", ".pq"}:
                return load_dataset("parquet", data_files=str(path))
            raise ValueError(
                f"Unsupported dataset file format for {dataset_path!r}. "
                "Expected a directory saved with `save_to_disk()` or a .json/.jsonl/.csv/.parquet file."
            )
        return load_from_disk(dataset_path)
    if dataset_config:
        return load_dataset(dataset_name, dataset_config)
    return load_dataset(dataset_name)


def resolve_split(dataset_obj, split_name: str):
    if hasattr(dataset_obj, "keys"):
        if split_name in dataset_obj:
            return dataset_obj[split_name]
        raise KeyError(
            f"Split '{split_name}' not found. Available splits: {list(dataset_obj.keys())}"
        )
    return dataset_obj
