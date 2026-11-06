# Algorithm conventions

Use the unified `climbing-gym train CONFIG.toml` entry point. These are complete
ported implementations; the methods retain their distinct reductions.

| `trainer.algorithm` | `advantage_estimator` | Example |
| --- | --- | --- |
| `ppo` | `gae` | `configs/train_ppo.toml` |
| `grpo` | `grpo` or `sa_mrpo` | `configs/train_grpo.toml` |
| `cispo` | `grpo` | `configs/train_cispo.toml` |
| `flashreinforce` | `batch_centered` | `configs/train_flashreinforce.toml` |
| `mis`, `mis_sc` | `batch_centered` | `configs/train_mis.toml`, `train_mis_sc.toml` |
| `grouped_mis`, `grouped_mis_sc` | `grpo` | `configs/train_grouped_mis*.toml` |
| `mis_ppo` | `value` with critic, `grpo` without | `configs/train_mis_ppo.toml` |
| `value_mis_sc` | `length_adaptive_gae` | `configs/train_value_mis_sc.toml` |

Native behavior probabilities are raw sampled-token log probabilities. Training
requires temperature 1, full support (`top_k=-1`, `top_p=1`), `n=1`, and
`logprobs=1`. The SC variants additionally request `behavior_topk` (1–128) in
sampling `extra_params`. Do not confuse this metadata with sampling top-k.

PPO uses clipped surrogate policy learning and a separate critic clone, terminal
outcome GAE, optional critic warmup, and independent optimizers. The historical
`grpo` estimator spelling remains a PPO alias for GAE; new examples use `gae`.
GRPO retains its group-relative estimator and packed reduction. CISPO clips
**detached importance weights**, allowing signed actor gradients beyond the clip
boundary; it requires `seq_mean_token_sum`, one policy epoch, and no auxiliary
entropy/format objective.

FlashREINFORCE centers outcomes with one batch mean, without reward standard
deviation. It gates complete responses using the sampled-action Bernoulli
KL(behavior || learner), computed in FP64. Admitted importance weights remain
unbounded. Its equation-11 reduction divides by original batch size and response
length; rejected responses do not change the baseline or denominator. Use direct
complete responses, one rollout per distinct prompt, `sample_mean`, and exactly
one accumulated update.

MIS masks out-of-bound token ratios before exponentiation. Ungrouped MIS uses a
fixed configured token normalizer; `legacy_response_mean` preserves the older
response-mean option explicitly. Sparse score centering uses exact sampler head
probabilities and the learner-proportional omitted tail. Weights and centering
coefficients are detached. This is token correction and does not establish exact
trajectory density correction.

Grouped MIS centers raw outcomes within complete mixed binary groups, without
standard deviation scaling, with global `B * max_tokens` normalization. It needs
dynamic sampling, at least two rollouts per prompt, binary rewards, and whole-group
request/concurrency budgets. Uniform groups can exhaust the request budget. Its
surplus queue stores immutable disk references with explicit policy-lag bounds.

MIS-PPO retains separate token and geometric-mean trajectory gates, optional
explicit fallback, active-token actor means, all-valid-token critic terms, and
detached value baselines. It uses one eager accumulated collection update and raw
terminal outcomes. Its actor reduction can depend on packing when token retention
differs between packs; that original convention is preserved. `mis_use_value=false`
selects the policy-only route and requires `advantage_estimator="grpo"`.

Value MIS+SC freezes actor advantages before fitting either model, using critic
pre-action positions, raw Monte Carlo targets, full-response token advantages,
and fixed configured token normalization. Reference-only critic context never
contains the current response/outcome. The cloned critic trains feed-forward/norm
parameters and a bounded FP32 value projection. Critic fit precedes actor update,
with alternating model residency and an independent optimizer. Mapped critic state
is disposable; checkpoints retain durable exact states.

Full numerical provenance, source revisions, and CPU verification are recorded
in [the algorithm port note](porting/rl_algorithms.md) and
[the runtime note](porting/training_runtime.md). These implementation checks do
not demonstrate convergence or learning gains on a long experiment.
