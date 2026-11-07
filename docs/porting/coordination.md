# Source provenance

Climbing Gym brings Lukas's SimplePT implementations into a standalone MIT-licensed package. Source revisions, selected checkpoint identities, and data hashes are retained so the algorithms and example recipes can be traced to their original implementations.

| Area | Source and implementation notes |
| --- | --- |
| Native inference and model kernels | [Native engine](native_engine.md), [source manifest](native_source_manifest.json) |
| Policy and value objectives | [Algorithm provenance](rl_algorithms.md), [training runtime](training_runtime.md) |
| Environments and collection | [Environment provenance](environments_rollouts.md), [CFOP recovery metadata](cfop.json) |
| SFT and projection | [SFT/projection](sft_projection.md), [historical Futoshiki projector](futoshiki_example_projection.md) |
| Reproducible climbs | [Futoshiki runtime](futoshiki_example_runtime.md), [Knights value method](knights_value_method.md) |
| Logging and evaluation | [Logging and trace contracts](logging_evaluation.md) |

Named source checkouts and historical artifact paths identify archived inputs; they are not runtime dependencies. Paths are relative to the original workspace, with machine-specific prefixes omitted. Commands in historical reports use portable `uv` equivalents of the migration environment's commands. Source and ported SHA256 fields in the native manifest describe the initial migration snapshot; subsequent fixes are recorded in Git history. Some original source commits and experiment archives are not publicly hosted.

Historical verification is recorded in the [native checkpoint and throughput report](release_verification.md), [follow-up review](followup_verification.md), and [package verification](followup_packaging.md). Files named under `artifacts/` in those reports are local QA outputs and are excluded from Git and distributions. The committed reports retain the measurements and reproduction commands; the examples include their data manifests and [complete recorded projection samples](../../examples/futoshiki_hill_climb/recorded_projection_samples.md).

Model weights remain external. Bundled data retain their source attribution and licenses; see the [puzzle bank source notice](../../examples/futoshiki_hill_climb/data/SOURCES.md).
