# Migration coordination

Destination: `/home/lukas928/ClimbingGym/climbing-gym`. Distribution `climbing-gym`; Python package `climbing_gym`.

Release license: MIT, explicitly selected by the user during the migration. Preserve existing source attribution; model weights retain their own licenses and are not included.

Repository visibility: create and validate locally only. The user explicitly requested no publishing yet; a later hosted repository should start private before becoming public. Do not create a remote, push, or publish artifacts during this task.

Baseline candidates are `/home/lukas928/SimplePT/simple-pt-memory-defaults` (recent memory defaults), `simple-pt-knights-value-mis-sc` (value/MIS/score centering), plus targeted worktrees discovered with `git worktree list` and path history. Source trees remain read-only. Port real implementations with renamed imports and no dependency on SimplePT checkout paths. Prefer focused modules with concise public entry points over copying historical scripts wholesale. Preserve useful comments and attribution.

Eight Sol-agent assignments, scheduled in waves because three worker slots are available:

1. Native inference and models: owns `src/climbing_gym/inference/`, `models/`, and related tests.
2. Environments and rollouts: owns `src/climbing_gym/envs/`, `rollout/`, and related tests. Only GSM8K, Countdown, Knights and Knaves, Futoshiki.
3. RL algorithms: owns algorithm modules (CISPO, FlashREINFORCE, PPO, MISPO, score centering and value support), and related tests.
4. Training integration and offload: owns shared training/runtime/configuration/data modules and trainer integration. Coordinate changes to algorithm modules.
5. SFT and projection/MCMC: owns SFT/sampling implementation and related tests/examples.
6. W&B and local logging: owns logging, evaluation trace/report helpers, logging integration/tests.
7. Release packaging and documentation: owns package metadata, CLI, README, configuration examples, CI, user docs, package root.
8. Independent compatibility/release verification: checks algorithm parity, imports, CPU suite, native LFM integration; owns verification reports and any agreed integration fixes.

Keep top-level algorithm module names initially if this preserves implementation clearly. Shared lower-level modules should retain the original relative layout when useful. Communicate cross-area dependencies rather than copying another owner's files. Avoid importing unsupported environments merely for registry completeness. Parent coordinates architectural decisions and integration.
