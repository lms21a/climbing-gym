# Contributing

Use Python 3.12 and [uv](https://docs.astral.sh/uv/) from a checkout of this
repository:

```bash
uv sync --locked --python 3.12 --group dev
CUDA_VISIBLE_DEVICES='' uv run pytest -q
uv build
```

Run the tests relevant to your change first. CUDA tests skip without a GPU.
For CUDA verification, follow the [acceleration setup](README.md#start-locally)
with a compatible NVIDIA GPU/toolkit, then run:

```bash
uv run --extra cuda pytest -q
```

Changes to native kernels or accelerator memory ownership need these CUDA
checks too. [The run guide](docs/running.md) explains native preflight, local
records, and complete model evaluation traces. Core development does not require
the access-restricted [CFOP solver](docs/cfop.md); its optional tests skip when
the bindings are unavailable.

Keep changes focused and preserve each algorithm's documented numerical
conventions. Include what changed and how it was checked. For numerical changes,
explain any intended difference from the existing behavior. Keep source
attribution and data licenses with redistributed material. Model weights,
credentials, private conversations, and generated runs belong outside commits.

The repository's [agent instructions](AGENTS.md) apply to coding agents too.
