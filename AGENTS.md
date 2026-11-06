# Climbing Gym contributor instructions

- Use `/home/lukas928/SimplePT/.venv/bin/python` for code and tests in this migration. Install software with `uv`.
- Keep the source SimplePT repositories unchanged. This is a standalone package named `climbing-gym`, imported as `climbing_gym`.
- Preserve working numerical behavior and checkpoint/model compatibility. Port implementations; do not replace them with algorithm sketches.
- Use only the native inference engine for inference and evaluation.
- Supported environments for this release: GSM8K, Countdown, Knights and Knaves, and Futoshiki.
- Preserve LFM 2.5 1.2B support, thinking tags, ordinary chat templates, rollout log probabilities, and memory/offload behavior.
- Before any model training/evaluation, validate inference batch/concurrency limits and benchmark representative native rollouts with the intended token budget and log probabilities; compare throughput and peak GPU memory with relevant prior benchmarks. Record throughput and full-step timing after launch. Speed guards are only for initial setup, never for stopping training during fluctuations.
- Every model evaluation must save actual prompts, complete responses, expected answers, and outcomes in a Markdown trace report, spanning available difficulty tiers and successes/failures.
- New training runs log online to W&B `lukas928/simple-pt`, and keep local metrics/checkpoints. No stronger-model distillation.
- Agents share this checkout. Respect file ownership in `docs/porting/coordination.md`; do not commit other agents' unfinished files. Report source paths and commits in your porting note.
