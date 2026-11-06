from types import SimpleNamespace

import pytest

from climbing_gym.numbered_notebook import NumberedNotebook
from climbing_gym.numbered_notebook import notebook_instruction
from climbing_gym.rollout.harnesses.numbered_notebook import NumberedNotebookHarness


def memory(**kwargs):
    return NumberedNotebook(lambda text: list(text), max_tokens=kwargs.pop('max_tokens', 1000), **kwargs)


def test_save_edit_delete_and_stable_automatic_numbers():
    m = memory()
    r = m.apply_turn('<save>A < B; tentative.</save><save>Try 3.</save></thinking><continue>')
    assert not r.errors and r.control == 'continue'
    assert m.entries == {1: 'A < B; tentative.', 2: 'Try 3.'}
    r = m.apply_turn('<edit 1>A=2; B uncertain.</edit><delete 2></delete><save>Check B.</save></thinking><continue>')
    assert not r.errors
    assert m.entries == {1: 'A=2; B uncertain.', 3: 'Check B.'}
    assert m.next_number == 4


def test_atomic_capacity_failures_do_not_consume_numbers():
    m = memory(max_tokens=12)
    m.apply('save', 'short')
    before = m.snapshot()
    for action in [('save', 'a' * 20, None), ('edit', 'a' * 20, 1), ('edit', 'oops', 2)]:
        with pytest.raises(ValueError):
            m.apply(*action)
        assert m.snapshot() == before


def test_entry_limit_and_trajectory_isolation():
    a, b = memory(max_entries=1), memory()
    a.apply('save', 'a')
    with pytest.raises(ValueError):
        a.apply('save', 'b')
    assert not b.entries and b.next_number == 1
    a.apply('delete', number=1)
    assert a.apply('save', 'c')['number'] == 2


@pytest.mark.parametrize('bad', [
    '<edit 0>x</edit>', '<edit 1>x</edit>', '<delete 1></delete>',
    '<save 1>x</save>', '<save></save>', '<delete 1>body</delete>',
    '<save><edit 1>nested</edit></save>',
])
def test_invalid_operations_leave_memory_unchanged(bad):
    m = memory()
    r = m.apply_turn(bad + '</thinking><continue>')
    assert r.errors
    assert m.entries == {} and m.next_number == 1


def test_length_cutoff_preserves_completed_updates_only():
    m = memory()
    r = m.apply_turn('<save>good</save><edit 1>unfinished', truncated=True)
    assert r.control == 'length' and r.incomplete_action
    assert m.entries == {1: 'good'}
    r = m.apply_turn('<save>truncated', truncated=False)
    assert r.errors and m.entries == {1: 'good'}


def test_terminal_answer_is_not_interpreted_as_memory():
    m = memory()
    r = m.apply_turn('Reasoning.<save>fact</save></thinking><answer>A=2</answer>')
    assert not r.errors and r.control == 'answer'
    assert r.response == 'Reasoning.</thinking><answer>A=2</answer>'
    r = m.apply_turn('</thinking><answer>A=2</answer><delete 1></delete>')
    assert r.errors and m.entries == {1: 'fact'}


def test_continue_must_end_message():
    m = memory()
    r = m.apply_turn('</thinking><continue><save>ignored</save>')
    assert r.errors and not m.entries


def test_instructions_show_only_valid_handles_without_placeholder_payloads():
    m=memory()
    def instructions(turn=1):
        return notebook_instruction(turn=turn,max_turns=3,turn_tokens=4096,notebook=m)
    empty=instructions()
    assert '<edit 1>' not in empty and '<delete 1>' not in empty
    assert 'literal note text' not in empty and 'replacement text' not in empty
    m.apply('save','old')
    m.apply('delete',number=1)
    m.apply('save','current')
    text=instructions()
    assert '<edit 2>' in text and '<delete 2></delete>' in text and '<edit 1>' not in text
    assert '<continue>' in text and '<continue>' not in instructions(turn=3)


def test_oversized_invalid_note_number_is_a_protocol_error():
    m = memory()
    r = m.apply_turn('<edit ' + '9'*5000 + '>text</edit></thinking><continue>')
    assert r.errors and not m.entries


class Tokenizer:
    eos_token_id = 2

    def encode(self, text, **kwargs):
        return list(text.encode())

    def apply_chat_template(self, messages, **kwargs):
        return messages[0]['content']


def result(text, reason='stop'):
    return SimpleNamespace(text=text, prompt='prompt', prompt_token_ids=(1,),
        outputs=[SimpleNamespace(token_ids=tuple(text.encode()), completion_logprobs=tuple([-.1]*len(text)),
                                 finish_reason=reason, stop_reason='stop')])


def test_harness_resets_scratchwork_and_injects_current_notebook():
    h = NumberedNotebookHarness(Tokenizer(), turn_tokens=4096, context_tokens=8192)
    example = SimpleNamespace(prompt='original puzzle')
    state = h.create_state(example, 0)
    h.apply_result(example, state, result('SECRET SCRATCH<save>visible note</save></thinking><continue>'))
    request = h.next_request(example, state)
    assert 'original puzzle' in request.prompt['prompt'] and '[1] visible note' in request.prompt['prompt']
    assert 'SECRET SCRATCH' not in request.prompt['prompt']
    assert request.sampling.logprobs == 1 and request.sampling.max_tokens == 4096
    h.apply_result(example, state, result('done</thinking><answer>A=2</answer>'))
    assert state.finished and state.terminal_reason == 'answer_submitted'
    assert len(state.turns) == 2 and len(state.turns[0].completion_logprobs) > 0


def test_turn_limit_and_context_limit():
    h = NumberedNotebookHarness(Tokenizer(), max_turns=1, context_tokens=8192)
    example = SimpleNamespace(prompt='puzzle')
    state = h.create_state(example, 0)
    h.apply_result(example, state, result('<save>keep</save>', 'length'))
    assert state.finished and state.terminal_reason == 'max_turns' and state.notebook.entries == {1: 'keep'}
    h.context_tokens = 10
    with pytest.raises(ValueError, match='exceed context'):
        h.next_request(example, h.create_state(example, 0))


def test_native_prompt_mapping_is_logged_as_text():
    h=NumberedNotebookHarness(Tokenizer())
    example=SimpleNamespace(prompt='task')
    state=h.create_state(example,0)
    output=result('done</thinking><answer>A=2</answer>')
    output.prompt={'prompt':'actual prompt text','prompt_token_ids':(1,)}
    h.apply_result(example,state,output)
    assert state.turns[0].prompt=='actual prompt text'


def test_config_factory_and_general_rollout_runner():
    import asyncio
    from climbing_gym.config.job import Harness
    from climbing_gym.envs import MathQAEnv, exact_match_math_reward
    from climbing_gym.rollout import RolloutRunner, RolloutRunnerConfig, RolloutConfig

    harness = Harness(name='numbered_notebook',args={'tokenizer':Tokenizer()}).build()
    assert isinstance(harness,NumberedNotebookHarness)
    env = MathQAEnv(reward_fn=exact_match_math_reward)
    env.load_rows([{'question':'What is 2 + 2?','parsed_answer':'4'}])

    class Engine:
        def __init__(self):
            self.prompts = []

        async def generate(self,prompt,**kwargs):
            self.prompts.append(prompt['prompt'])
            text = ('PRIVATE WORK<save>Partial result: 4.</save></thinking><continue>' if len(self.prompts)==1
                    else '<edit 1>Checked result: 4.</edit></thinking><answer>\\boxed{4}</answer>')
            return result(text)

    engine = Engine()
    trajectories = asyncio.run(RolloutRunner(RolloutRunnerConfig(effective_batch_size=1)).collect(
        env=env,engine=engine,harness=harness,rollout_config=RolloutConfig(num_rollouts=1)))
    trajectory = trajectories[0]
    assert len(trajectory.turns)==2 and trajectory.evaluation.reward.reward==1.
    assert '[1] Partial result: 4.' in engine.prompts[1] and 'PRIVATE WORK' not in engine.prompts[1]
    assert trajectory.turns[-1].metadata['memory_after']['entries']=={'1':'Checked result: 4.'}
