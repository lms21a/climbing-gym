# Follow-up packaging and documentation

This integrates the complete Futoshiki hill-climb example, the recent Knights
Value MIS+SC method, optional CFOP, and the requested removal of GitHub workflows.
The baseline is local commit `0bf6766`. Implementation/source provenance lives in
[the Futoshiki runtime note](futoshiki_example_runtime.md),
[projection note](futoshiki_example_projection.md),
[Knights method note](knights_value_method.md), and [CFOP metadata](cfop.json).
Original SimplePT trees remained read-only; the package has no runtime imports
from them. All Python checks used `/home/lukas928/SimplePT/.venv/bin/python`
with `CUDA_VISIBLE_DEVICES=''`; dependency/build/install operations used `uv`.

## User documentation and examples

README links both runnable examples, CFOP setup, the Futoshiki data/source notice,
and complete recorded projection samples. It distinguishes historical results
and the initial native GPU verification from the follow-up CPU checks. The run
and algorithm guides describe CFOP's separate phase-program format and optional
binding, Value MIS+SC's BPCO-style privileged critic/length-adaptive advantages,
and readiness-only critic warmup. The Knights example documents production and
smoke profiles, historical initialization, exact resume, retention, and held-out
panels. No unrelated BPO CLI was introduced.

The workflow file and now-empty `.github` directories were removed. Sdist
selection excludes `.github`; local test/build commands remain. Historical notes
identify the removed configuration as history. No current CI promise or account
setting change is part of this update.

Seven user-facing documents were checked for local links and shell syntax in
all Bash fences. Actual help commands for Futoshiki's preparation/projection/RL
scripts and Knights' preparation/runner scripts succeed on CPU. Standalone
configuration tests cover all ten method routes and the optional CFOP examples;
CFOP's two-row examples no longer inherit the older three-row assertion. Tests
skip only CFOP construction when its optional binding is absent.

## Contents and provenance

The source distribution intentionally includes **48 compressed JSONL assets**,
**54,340,976 bytes** total (51.82 MiB), across the two examples. The largest asset
is 17,030,356 bytes (16.24 MiB). Futoshiki contributes 45 assets/53,987,850 bytes;
Knights contributes three/353,126 bytes. Banks/manifests preserve exact source
identities and hashes; model responses are explicitly labeled recorded task
supervision and traces. The GSM8K source/license attribution is retained.
Historical machine paths are provenance metadata, not executable dependencies.

The wheel contains code and package metadata. The sdist additionally contains
examples, compressed data, tests, configurations, docs, and the lockfile. Neither
contains weights, private conversation transcripts, original source snapshots,
experiment directories, Python caches, or GitHub workflows. Optional private
`cfop-solver` bindings are not a base dependency or lockfile source; installing
and using other environments does not require them. Model weights remain
user-provided and separately licensed.

## Final verification

The full repository CPU suite passed **673 tests**, with **130 CUDA skips**,
**14 existing TorchScript deprecation warnings**, in **26.31 seconds**:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=8 PYTHONPATH=src \
  /home/lukas928/SimplePT/.venv/bin/python -m pytest -q
uv lock --check --python /home/lukas928/SimplePT/.venv/bin/python
uv build --python /home/lukas928/SimplePT/.venv/bin/python
```

The focused packaging suite passed **26 tests**. The lockfile check succeeds.
Final distribution validation compares every current source/test/config/doc/
example byte with the sdist and every Python source byte with the wheel. An
isolated no-dependency wheel install into a temporary target uses the mandated
interpreter from `/tmp`, imports **21 API modules** from that wheel, runs all
**six command help pages** and the console version, and confirms that blocking
the optional CFOP solver still permits other environments and module imports.
All **five example help commands** run from an extracted sdist against the
installed wheel, without using the repository's Python source path.

Local machine-readable sizes, hashes, temporary install paths, and distribution
results are in `artifacts/packaging/validation.json`. Full CPU output is
`artifacts/packaging/followup_cpu.txt`; static documentation/data inventory is
`artifacts/packaging/followup_static.json`. These remain ignored local evidence.
Build outputs are `dist/climbing_gym-0.1.0-py3-none-any.whl` and
`dist/climbing_gym-0.1.0.tar.gz`.

No GPU model inference, evaluation, or training was launched by this packaging
work. Tiny policies are unit-test fixtures. This establishes package/continuation
contracts and example availability; it does not establish full-model capacity,
convergence, throughput, or historical accuracy reproduction for the new runs.
See [the independent follow-up review](followup_verification.md) and
[recorded complete model samples](../../examples/futoshiki_hill_climb/recorded_projection_samples.md)
for the corresponding scope and actual archived outputs.
