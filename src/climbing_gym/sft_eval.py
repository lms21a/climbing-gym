"""Periodic native evaluation for offline SFT using the standalone eval API."""
from dataclasses import dataclass, replace
from pathlib import Path
import time

import torch

from .eval import load_run_from_toml, run_eval


@dataclass(frozen=True, slots=True)
class SFTEvaluationConfig:
    config_file: str = ''
    interval_steps: int = 0

    def __post_init__(self):
        if bool(self.config_file) != (self.interval_steps > 0):
            raise ValueError('SFT [eval] requires config_file and a positive interval_steps together')


def _move_optimizer(optimizer, device):
    # SFT's AdamW state consists of tensors and scalar entries.
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def _cleanup_resources(actions, primary_error=None):
    """Attempt every cleanup, retaining the failure that triggered cleanup."""
    failures = []
    for label, action in actions:
        try:
            action()
        except BaseException as error:
            failures.append((label, error))
    if primary_error is not None:
        for label, error in failures:
            primary_error.add_note(f"Cleanup of {label} also failed: {type(error).__name__}: {error}")
    elif failures:
        label, error = failures[0]
        error.add_note(f"Failure occurred while cleaning up {label}")
        for later_label, later_error in failures[1:]:
            error.add_note(f"Cleanup of {later_label} also failed: {type(later_error).__name__}: {later_error}")
        raise error


class PeriodicSFTEvaluator:
    """Load/validate preflight once, then reuse native engine and dataset.

    Call construction while trainables are on CPU. Between evaluations the
    native actor retains weights but releases KV/graph buffers. No inference
    speed guard or probe runs during training.
    """
    def __init__(self, config: SFTEvaluationConfig, *, model, optimizer, train_device, output_dir):
        self.config, self.model, self.optimizer = config, model, optimizer
        self.train_device, self.output_dir = train_device, Path(output_dir)
        # Standalone loading validates its saved workload-matched preflight.
        owned_run = load_run_from_toml(config.config_file)
        try:
            self.run = replace(owned_run, close_engine_on_finish=False)
            self.run.engine.release_inference_buffers()
        except BaseException as error:
            _cleanup_resources([('native evaluation engine', owned_run.engine.close)], error)
            raise

    def evaluate(self, step: int) -> dict:
        if (step+1) % self.config.interval_steps:
            return {}
        started = time.perf_counter()
        was_training = self.model.training
        failure = None
        try:
            self.model.to('cpu')
            _move_optimizer(self.optimizer, 'cpu')
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.run.engine.sync_from_model(self.model)
            directory = self.output_dir/f'step_{step}'
            logging = replace(self.run.logging_config, summary_path=str(directory/'summary.json'),
                trajectory_path=str(directory/'trajectories.jsonl'), trace_path=str(directory/'traces.md'))
            metrics = run_eval(replace(self.run, logging_config=logging))
            return {**{f'eval/{key}': float(value) for key, value in metrics.items()},
                    'eval/full_step_seconds': time.perf_counter()-started}
        except BaseException as error:
            failure = error
            raise
        finally:
            _cleanup_resources([
                ('native evaluation buffers', self.run.engine.release_inference_buffers),
                ('SFT model device', lambda: self.model.to(self.train_device)),
                ('SFT optimizer device', lambda: _move_optimizer(self.optimizer, self.train_device)),
                ('SFT model mode', lambda: self.model.train(was_training)),
            ], failure)

    def close(self):
        self.run.engine.close()
