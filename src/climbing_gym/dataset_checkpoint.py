"""Dataset continuation at complete rollout boundaries, independent of optimizer state."""
import json
from pathlib import Path
import warnings


def save_dataset_checkpoint(coordinator, checkpoint_dir):
    checkpoint_dir = Path(checkpoint_dir)
    runner = coordinator.rollout_runner
    # Generic multi-turn runners may carry unfinished inference state. Do not
    # pretend that a cursor alone can resume those trajectories faithfully.
    if getattr(runner, '_unfinished_groups', ()):
        warnings.warn('Dataset state not saved: unfinished rollout groups remain')
        return
    if getattr(coordinator.env, '_dataset_cursor', None) is None:
        return
    payload = {'version': 1, 'env': coordinator.env.dataset_state_dict(),
               'next_group_id': runner._next_group_id,
               'update_index': getattr(runner, 'update_index', 0)}
    engine = getattr(coordinator, 'engine', None)
    if hasattr(engine, '_generation_index'):
        # Collections have drained at this boundary. Keep the native seed
        # sequence advancing across restart rather than reusing initial seeds.
        with engine._seed_lock:
            payload['native_generation_index'] = int(engine._generation_index)
    path = checkpoint_dir / 'dataset_state.json'
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload, indent=2) + '\n')
    temporary.replace(path)


def restore_dataset_checkpoint(coordinator, checkpoint_dir):
    checkpoint_dir = Path(checkpoint_dir)
    path = checkpoint_dir / 'dataset_state.json'
    if not path.exists():
        warnings.warn('Checkpoint has no dataset state; starting a fresh dataset cursor')
        return False
    payload = json.loads(path.read_text())
    if payload['version'] != 1:
        raise ValueError('Unsupported dataset checkpoint version')
    coordinator.env.load_dataset_state_dict(payload['env'])
    runner = coordinator.rollout_runner
    runner._next_group_id = int(payload['next_group_id'])
    if hasattr(runner, 'update_index'):
        runner.update_index = int(payload['update_index'])
    engine = getattr(coordinator, 'engine', None)
    if 'native_generation_index' in payload and hasattr(engine, '_generation_index'):
        index = payload['native_generation_index']
        if type(index) is not int or index < 0:
            raise ValueError('Invalid native generation seed counter')
        with engine._seed_lock:
            engine._generation_index = index
    return True
