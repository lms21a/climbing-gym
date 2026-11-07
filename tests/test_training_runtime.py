"""CPU integration checks; no pretrained inference, GPU learning, or evaluation."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import json
import random

import pytest
import torch

from climbing_gym.config import ConfigManager, Env, Inference, PPOTrainer
from climbing_gym.cpu_master_adamw import CPUStateMasterAdamW
from climbing_gym.grpo import ResolvedGRPORun, _build_rollout_engine
from climbing_gym.ppo import ResolvedPPORun, MasterWeightAdamW, build_value_model_from_policy
from climbing_gym.rollout import TrajectoryBatch
from climbing_gym.train import resolve_config, validate_config, update_policy, run_training
from climbing_gym.trainers.checkpoints import CheckpointManager, CheckpointManagerConfig, restore_training_state
from climbing_gym.trainers.common import collect_rollout_batch_with_trainables
from rl_fixtures import TinyPolicy
from test_flashreinforce import config
from test_offpolicy import make_rows
from test_value_mis_sc import TinyBackbone, Tokenizer


class Policy(TinyPolicy):
    def forward(self, inputs, positions=None):
        return self.lm_head(self.get_hidden_states(inputs, positions))

    def save_as_hf(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), Path(path) / 'weights.pt')


class Engine:
    def __init__(self):
        self.events = []

    def release_inference_buffers(self):
        self.events.append('release')

    def close(self):
        self.events.append('close')


class Coordinator:
    def __init__(self, engine, rows):
        self.engine, self.rows = engine, rows
        self.env = SimpleNamespace(_dataset_cursor=None)
        self.rollout_runner = SimpleNamespace(_unfinished_groups=[])

    def sync_weights(self, model, **kwargs):
        self.engine.events.append('sync')

    def collect(self):
        self.engine.events.append('collect')
        return TrajectoryBatch(indices=tuple(range(len(self.rows))), trajectories=tuple(self.rows), stats={})


class Logger:
    def __init__(self):
        self.metrics = []

    def log_step(self, **kwargs):
        self.metrics.append(kwargs)

    def close(self):
        pass


def run_fixture(cfg, policy=None):
    policy = policy or Policy()
    engine = Engine()
    return ResolvedGRPORun(policy_model=policy, policy_optimizer=torch.optim.SGD(policy.parameters(), lr=.1),
        rollout_coordinator=Coordinator(engine, make_rows(policy)), config=cfg, engine=engine,
        checkpoint_manager=CheckpointManager(CheckpointManagerConfig()), wandb_logger=Logger())


@pytest.mark.parametrize('algorithm', ['flashreinforce', 'mis', 'mis_sc', 'mis_ppo', 'grouped_mis', 'grouped_mis_sc'])
def test_one_native_runtime_collection_and_real_cpu_update(algorithm):
    torch.manual_seed(30)
    cfg = config(flashreinforce_fused_head=False, flashreinforce_delta=100.)
    cfg = replace(cfg, trainer=replace(cfg.trainer, algorithm=algorithm,
        mis_token_ratio_min=.1, mis_token_ratio_max=10., mis_trajectory_ratio_min=.1, mis_trajectory_ratio_max=10.))
    if algorithm.endswith('mis_sc'):
        cfg = replace(cfg, inference=replace(cfg.inference, sampling=replace(cfg.inference.sampling,
                      extra_params={'behavior_topk':16})))
    if algorithm.startswith('grouped'):
        cfg = replace(cfg, trainer=replace(cfg.trainer, advantage_estimator='grpo'),
            rollout=replace(cfg.rollout, num_rollouts=3, batch_size=1, effective_batch_size=1,
                            max_total_requests=3, max_inflight_requests=3, use_dynamic_sampling=True))
    if algorithm == 'mis_ppo':
        cfg = replace(cfg, trainer=replace(cfg.trainer, advantage_estimator='value'))
    cfg = resolve_config(cfg)
    validate_config(cfg)
    run = run_fixture(cfg)
    if algorithm == 'mis_ppo':
        critic = build_value_model_from_policy(run.policy_model)
        run = replace(run, value_model=critic, value_optimizer=torch.optim.SGD(critic.parameters(), lr=.1))
    if algorithm.startswith('grouped'):
        rows = run.rollout_coordinator.rows
        run.rollout_coordinator.rows = [replace(row, group_id=0, rollout_index=i,
            example=rows[0].example) for i, row in enumerate(rows)]
    before = deepcopy(run.policy_model.state_dict())
    run_training(run)
    assert any(not torch.equal(before[name], value) for name, value in run.policy_model.state_dict().items())
    assert run.engine.events.index('collect') < run.engine.events.index('release')
    assert run.engine.events[-1] == 'close'
    metrics = run.wandb_logger.metrics[0]['metrics']
    assert metrics['trainer/rollout_tokens_per_second'] > 0
    assert metrics['trainer/full_step_time_seconds'] > 0


def test_value_mis_runtime_freezes_targets_fits_critic_and_updates_actor():
    from climbing_gym.value_mis_sc import build_critic
    torch.manual_seed(14)
    policy = TinyBackbone()
    cfg = config(flashreinforce_fused_head=False, value_epochs=1)
    cfg = resolve_config(replace(cfg, trainer=replace(cfg.trainer, algorithm='value_mis_sc', advantage_estimator='length_adaptive_gae')))
    run = run_fixture(cfg, policy)
    critic = build_critic(policy)
    run = replace(run, value_model=critic, value_optimizer=torch.optim.SGD(critic.parameters(), lr=.1), tokenizer=Tokenizer())
    actor_before, critic_before = deepcopy(policy.state_dict()), deepcopy(critic.state_dict())
    run_training(run)
    assert any(not torch.equal(actor_before[n], p) for n, p in policy.state_dict().items())
    assert any(not torch.equal(critic_before[n], p) for n, p in critic.state_dict().items())
    assert all(p.dtype == torch.float32 for p in critic.value_head.parameters())
    assert run.wandb_logger.metrics[0]['metrics']['value/optimizer_steps'] == 1


@pytest.mark.parametrize('algorithm', ['ppo', 'cispo', 'grpo'])
def test_original_full_trainer_executes_a_cpu_optimizer_step(monkeypatch, algorithm):
    from climbing_gym import ppo, grpo
    cfg = config(flashreinforce_fused_head=False)
    cfg = replace(cfg, trainer=replace(cfg.trainer, algorithm=algorithm, advantage_estimator='grpo',
        loss_aggregation='seq_mean_token_sum' if algorithm == 'cispo' else 'token_mean',
        behavior_parity_max_mean_abs_probability_shift=1.))
    run = run_fixture(cfg)
    if algorithm != 'ppo':
        rows = run.rollout_coordinator.rows
        run.rollout_coordinator.rows = [replace(row, group_id=0, rollout_index=i) for i,row in enumerate(rows)]
    before = deepcopy(run.policy_model.state_dict())
    if algorithm == 'ppo':
        critic = build_value_model_from_policy(run.policy_model)
        fields = {name: getattr(run, name) for name in ResolvedPPORun.__dataclass_fields__ if hasattr(run, name)}
        fields.update(value_model=critic, value_optimizer=torch.optim.SGD(critic.parameters(), lr=.1))
        run = ResolvedPPORun(**fields)
        module = ppo
    else:
        module = grpo
        monkeypatch.setattr(grpo.ConfigManager, 'from_toml', lambda path: SimpleNamespace(config=cfg))
    monkeypatch.setattr(module, 'load_run_from_toml', lambda path: run)
    assert module.main(['unused.toml']) == 0
    assert any(not torch.equal(before[name], value) for name,value in run.policy_model.state_dict().items())
    assert run.engine.events.index('collect') < run.engine.events.index('release')


def test_native_dispatch_never_constructs_an_external_engine(monkeypatch):
    from climbing_gym import grpo
    captured = []
    monkeypatch.setattr(grpo, 'LFM25AsyncInferenceEngine', lambda *args, **kwargs: captured.append((args,kwargs)) or 'native')
    cfg = config()
    cfg = replace(cfg, inference=replace(cfg.inference, model_path='/portable/model', native_max_batch_size=64))
    assert _build_rollout_engine(cfg) == 'native'
    assert captured[0][1]['max_batch_size'] == 64
    with pytest.raises(ValueError, match='native'):
        Inference(backend='vllm')


def test_release_happens_before_trainables_reload_on_collection_failure(monkeypatch):
    from climbing_gym.trainers import common
    events = []
    model = torch.nn.Linear(2,2)
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    engine = SimpleNamespace(release_inference_buffers=lambda: events.append('release'))
    def fail():
        events.append('collect')
        raise RuntimeError('collection failed')
    coordinator = SimpleNamespace(engine=engine, sync_weights=lambda *a,**k: events.append('sync'), collect=fail)
    monkeypatch.setattr(common, 'offload_trainables_to_cpu', lambda trainables: events.append('offload'))
    monkeypatch.setattr(common, 'reload_trainables_to_device', lambda *a,**k: events.append('reload'))
    with pytest.raises(RuntimeError, match='collection failed'):
        collect_rollout_batch_with_trainables(rollout_coordinator=coordinator, trainables=[(model,optimizer)],
                                              sync_model=model, device='cpu')
    assert events == ['offload','sync','collect','release','reload']


def test_exact_checkpoint_after_offload_restores_master_moments_and_rng(tmp_path):
    from climbing_gym.trainers.common import offload_to_cpu, reload_to_device
    torch.manual_seed(3)
    random.seed(9)
    model = Policy().to(torch.bfloat16)
    optimizer = MasterWeightAdamW(model.parameters(), lr=1e-3, weight_decay=.01)
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    offload_to_cpu(model, optimizer)
    manager = CheckpointManager(CheckpointManagerConfig(root_dir=str(tmp_path)))
    path = manager.save(step=3, policy_model=model, policy_optimizer=optimizer)
    expected_torch, expected_python = torch.rand(3), random.random()
    resumed = deepcopy(model)
    resumed_optimizer = MasterWeightAdamW(resumed.parameters(), lr=1e-3, weight_decay=.01)
    assert restore_training_state(path, policy_optimizer=resumed_optimizer)
    assert torch.equal(torch.rand(3), expected_torch)
    assert random.random() == expected_python
    reload_to_device(model, optimizer, device='cpu')
    reload_to_device(resumed, resumed_optimizer, device='cpu')
    for a,b in zip(model.parameters(), resumed.parameters()):
        a.grad = torch.ones_like(a) * .3
        b.grad = a.grad.clone()
    optimizer.step(); resumed_optimizer.step()
    for a,b in zip(model.parameters(), resumed.parameters()):
        assert torch.equal(a,b)
        assert optimizer.state[a]['master_param'].dtype == torch.float32
        assert torch.equal(optimizer.state[a]['master_param'],resumed_optimizer.state[b]['master_param'])
    with pytest.raises(ValueError, match='no optimizer state'):
        restore_training_state(tmp_path/'legacy', policy_optimizer=resumed_optimizer)
    assert not restore_training_state(tmp_path/'legacy', policy_optimizer=resumed_optimizer, allow_weights_only=True)


def test_ppo_cpu_state_builder_obeys_bounded_workspace():
    from climbing_gym.ppo import _build_policy_optimizer, _build_value_optimizer_from_config
    trainer = PPOTrainer(optimizer_chunk_elements=2**31-1, optimizer_workspace_bytes=1025)
    for builder in (_build_policy_optimizer, _build_value_optimizer_from_config):
        optimizer = builder(Policy(), {'type':'cpu_master_adamw'}, trainer=trainer)
        assert isinstance(optimizer, CPUStateMasterAdamW)
        assert optimizer.chunk_elements == 85 and optimizer.workspace_bytes == 1020


@pytest.mark.parametrize('wandb_settings,expected_entity,expected_project', [
    ('', None, 'climbing-gym'),
    ("[wandb]\nentity='   '\nproject=''\n", None, 'climbing-gym'),
    ("[wandb]\nentity='example-team'\nproject='custom-project'\n", 'example-team', 'custom-project'),
])
def test_portable_paths_and_online_wandb_defaults(tmp_path, monkeypatch, wandb_settings, expected_entity, expected_project):
    location = tmp_path/'configs'
    location.mkdir()
    path = location/'run.toml'
    path.write_text('''[model]
name='lfm2'
path='../weights'
[data]
dataset_path='../data/train.jsonl'
[env]
name='gsm8k'
[inference]
preflight_report_path='../reports/native.json'
[rollout]
use_dynamic_sampling=false
[checkpoint]
root_dir='../checkpoints'
''' + wandb_settings)
    monkeypatch.chdir(tmp_path)
    cfg = ConfigManager.from_toml(path).config
    assert cfg.model.path == str(tmp_path/'weights')
    assert cfg.data.dataset_path == str(tmp_path/'data/train.jsonl')
    assert cfg.checkpoint.root_dir == str(tmp_path/'checkpoints')
    assert cfg.inference.preflight_report_path == str(tmp_path/'reports/native.json')
    assert cfg.wandb.enabled and cfg.wandb.mode == 'online'
    assert cfg.wandb.project == expected_project and cfg.wandb.entity == expected_entity
    assert cfg.inference.backend == 'lfm25_native'
    assert type(cfg.env.build()).__name__ == 'MathQAEnv'
    with pytest.raises(ValueError, match='Unsupported'):
        Env(name='mastermind').build()
