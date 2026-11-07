# Climbing Gym contributor instructions

- Use Python 3.12 and `uv`. Install development dependencies with `uv sync --locked --python 3.12 --group dev`, and run code/tests through the project environment with `uv run`.
- Keep `climbing-gym` (`climbing_gym`) standalone; do not add dependencies on local research checkouts or machine-specific paths.
- Preserve numerical behavior, checkpoint/tokenizer compatibility, and LFM 2.5 Base/Instruct/Thinking support. Keep existing source attribution and document intentional changes to algorithm conventions.
- Use only the native inference engine for inference and evaluation. Core environments are GSM8K, Countdown, Knights and Knaves, and Futoshiki. CFOP is an optional adapter requiring separately obtained solver bindings; see `docs/cfop.md`.
- Before model training/evaluation, benchmark representative native requests with the intended token budget, log probabilities, batch size and all concurrency limits. Compare throughput and peak GPU memory to a relevant prior or record an initial baseline. Inspect rollout/full-step timings after launch; do not add training-time throughput watchdogs or automatic speed probes.
- Every model evaluation must save complete actual prompts/responses, expected answers, and outcomes in a Markdown trace report, covering available difficulty and success/failure strata.
- New training runs default to online W&B project `climbing-gym` under the current account; retain local metrics and checkpoints. Allow explicit account/project overrides and deliberate offline/disabled logging.
- Use the current model, algorithmic solvers/verifiers, and generated data. Do not introduce stronger-model distillation.
- Keep weights, credentials, private conversations, caches, and generated run artifacts out of commits. Preserve recorded scientific results and bundled data attribution.
- Run focused CPU checks for changes; use CUDA tests when changing native kernels or accelerator lifecycle behavior. See `CONTRIBUTING.md` for local commands. Do not launch model workloads merely to edit documentation.
