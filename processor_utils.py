"""Filesystem rules shared by the training and inference processors."""
from pathlib import Path


def ensure_separate_output(output, *inputs):
    """Reject overlapping input/output trees before any output is written."""
    output = Path(output).resolve()
    for source in inputs:
        if source is None:
            continue
        source = Path(source).resolve()
        if source == output or source in output.parents or output in source.parents:
            raise ValueError(f"Input and output paths must not overlap: {source}, {output}")


def prepare_output(output, allow_cache=False):
    """Accept a pre-created empty output directory, never overwrite old results."""
    output = Path(output)
    if output.exists():
        existing = [p for p in output.iterdir() if not (allow_cache and p.name == '.cache')]
        if existing:
            raise FileExistsError(f"Output is not empty; choose a fresh directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
