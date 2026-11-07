# Environments and rollouts port

## Provenance

Ported working Python implementations from read-only `simple-pt-memory-defaults`, commit `6edae8f` (`codex/native-training-memory-defaults-20261104`). Paths under `src/simple_pt_2/` map directly to `src/climbing_gym/`:

- `envs/{contracts,countdown,futoshiki,knights_knaves,math_qa}.py` and a narrowed `envs/__init__.py`.
- `rollout/{config,contracts,coordinator,overlap,runner}.py`, public exports, and `harnesses/{direct,infinity_think,numbered_notebook}.py`.
- Independent solver/verifier and generation helpers: `countdown.py`, `countdown_answer_audit.py`, `doomloop.py`, `futoshiki.py`, `knights_knaves.py`, `knights_knaves_data.py`, `numbered_notebook.py`.
- `data/{dataset,__init__}.py`, temporarily assigned to this agent to unblock verification; ownership passes to training integration at handoff.

Inspected `git worktree list` and targeted path history. Compared the current baseline against Knights value/MIS/SC (`07b6762`), paragraph credit (`a7bca6d`), Futoshiki first-expert optimized (`089ab02`), and tier-two optimized (`93a99c3`). All five contain identical Futoshiki solver and rollout runner blobs (`efd5d8ad870eaa962a4d5663289b70084dc4cee4` and `e352e902237309ca66309b439d58d609cc2d6bcf`). The recent Knights/paragraph branches contain no newer core environment/rollout changes; optimized Futoshiki branches predate the supported Knights adapter. Their experiment scripts were not imported. The Knights environment entered the source history in `3522bfd`.

No source checkout was modified. No commits were made.

## Public behavior and integration

The environment package exports exactly four task families: GSM8K-style math (`MathQAEnv`, compatibility name `math_qa`), Countdown, Knights and Knaves, and Futoshiki. There are no Mastermind, rule-chain, string-reversal, Sudoku, or split-relay imports. The configuration registry must enforce the same four-family boundary.

The port preserves numerical reward/scoring behavior, including strict terminal answer blocks; Countdown's legal-integer arithmetic and alternative-solution verifier; optional terminal-expression and doom-loop rewards; Futoshiki partial-credit profiles and exact-match mode; and Knights exhaustive world enumeration, operator semantics, solver certificates, malformed-answer rejection, and deterministic tiered generation. Cached prompt/solution data does not replace the Knights public solver.

Math handles canonical GSM8K `question` / `answer` rows (extracting the gold value after `####`) and existing `parsed_answer` rows. Ordinary boxed answers, answer tags, prefilled `<thinking>` continuation validation, and multi-turn format/reward helpers retain existing semantics.

`RolloutCoordinator.collect()` is the synchronous bridge; `collect_async()` supports active event loops. `RolloutRunner` retains bounded asynchronous requests, abort/drain behavior, dynamic sampling, prefetched groups, request budgets, stall diagnostics, and group identity. Direct, InfinityThink, and numbered-notebook harnesses remain; experimental Countdown relay/compaction harness families are intentionally omitted. `one_batch_ahead` retains bounded producer/learner overlap and independently synchronized inference snapshots.

Native integration requires `inference.config.SamplingConfig`, `inference.types.BehaviorTopK`, native `generate`, abort, and model-sync methods. Token IDs, complete native completion log probabilities, behavior top-k tensors, group IDs, rollout indices, dataset indices, and trajectory metadata survive conversion. The algorithm agent confirmed these contracts cover value/MIS/score-centering requirements, including `policy_objective_weight` metadata. The notebook harness loads tokenizers through `climbing_gym.models.lfm2_compat`.

`examples/data/` contains three rows per supported family. Math examples are handwritten synthetic arithmetic in GSM8K's schema, not benchmark examples or measurements. Countdown, Knights, and Futoshiki examples were generated with the ported algorithmic generators; they include multiple available difficulty tiers.

## Verification

Used only `uv run python`, with the destination `src/` on `PYTHONPATH`. **180 CPU tests passed in 5.69 seconds** across environment/reward parsing, independent arithmetic and Boolean semantics, Futoshiki Latin-square enumeration and uniqueness, deterministic/disjoint datasets, solver certificate tampering, malformed answers, notebook operations, direct/multi-turn rollouts, bounded async requests, aborts, stall detection, sync/async coordinator calls, overlap, and dataset cursor state. Native sampler metadata preservation has dedicated tests, including the distinction between unavailable and empty log probabilities.

Ported source tests were narrowed by removing unrelated environments and script-only historical experiments. Two source tests were stale relative to the implementation; production behavior was not changed:

1. The InfinityThink test expected only three metadata keys. It now asserts the preserved summary/history values and strict intermediate/final format flags that the source harness already emits.
2. The math hybrid reward test expected obsolete additive shaping (`-0.2`, `1.4`, `-1.05`). The source function already computes binary `correct × strict_format_ok`; the test now checks valid strict metadata, incorrect answers, and missing format metadata against that existing rule.

Configuration-dependent tests retain local `climbing_gym.config` imports and must be run after the training/configuration wave lands. They were excluded from the 180-test run because the configuration package did not yet exist. Test files are `test_{envs,rollout,rollout_provenance,rollout_overlap,data,countdown,countdown_answer_audit,futoshiki,knights_knaves,numbered_notebook,math_single_answer_reward,doomloop}.py`. The independent release verifier should rerun all of them after integration.

No GPU inference, model evaluation, or training was launched. Consequently there is no claimed native throughput or model-accuracy result, and no newly generated model trace report. Before a real run, perform the mandated native-engine batch/concurrency/throughput/memory checks, then retain runtime timing logs and actual complete evaluation traces.
