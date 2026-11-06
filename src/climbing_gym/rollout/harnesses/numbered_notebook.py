"""Task-independent bounded turns with a model-edited numbered notebook."""
from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Mapping

from climbing_gym.inference.config import SamplingConfig
from climbing_gym.numbered_notebook import NumberedNotebook, render_notebook_prompt
from climbing_gym.rollout.contracts import HarnessRequest, HarnessState, build_trajectory_turn


@dataclass
class NotebookState(HarnessState):
    notebook: NumberedNotebook | None = None
    memory_errors: list[str] = field(default_factory=list)


class NumberedNotebookHarness:
    def __init__(self, tokenizer=None, *, tokenizer_path=None, turn_tokens=4096, memory_tokens=1024,
                 max_entries=32, max_turns=4, context_tokens=8192, temperature=1., behavior_topk=0):
        if min(turn_tokens, memory_tokens, max_entries, max_turns, context_tokens) <= 0:
            raise ValueError('Harness limits must be positive')
        if tokenizer is None:
            if not tokenizer_path:
                raise ValueError('Provide a tokenizer or tokenizer_path')
            from climbing_gym.models.lfm2_compat import load_lfm2_tokenizer
            tokenizer = load_lfm2_tokenizer(str(tokenizer_path))
        self.tokenizer = tokenizer
        self.turn_tokens, self.memory_tokens = turn_tokens, memory_tokens
        self.max_entries, self.max_turns = max_entries, max_turns
        self.context_tokens, self.temperature = context_tokens, temperature
        if type(behavior_topk) is not int or not 0 <= behavior_topk <= 128:
            raise ValueError('behavior_topk must be an integer from 0 to 128')
        self.behavior_topk = behavior_topk

    def create_state(self, example, rollout_index):
        return NotebookState(rollout_index=rollout_index,
            notebook=NumberedNotebook(lambda t: self.tokenizer.encode(t, add_special_tokens=False),
                                     max_tokens=self.memory_tokens, max_entries=self.max_entries),
            metadata={'harness': 'numbered_notebook'})

    def next_request(self, example, state):
        if state.finished:
            return None
        text = render_notebook_prompt(self.tokenizer, example.prompt, state.notebook,
            turn=len(state.turns)+1, max_turns=self.max_turns, turn_tokens=self.turn_tokens,
            feedback=state.memory_errors)
        ids = tuple(self.tokenizer.encode(text, add_special_tokens=False))
        if len(ids) + self.turn_tokens > self.context_tokens:
            raise ValueError('Task, notebook and full turn budget exceed context limit')
        return HarnessRequest(prompt=dict(prompt=text, prompt_token_ids=ids), sampling=SamplingConfig(
            temperature=self.temperature, top_p=1., top_k=-1, max_tokens=self.turn_tokens,
            logprobs=1, stop=('<continue>', '</answer>'),
            extra_params={'behavior_topk':self.behavior_topk} if self.behavior_topk else {}))

    def apply_result(self, example, state, result):
        output = result.outputs[0]
        if len(output.token_ids) > self.turn_tokens:
            raise ValueError('Generated response exceeded the turn budget')
        before = state.notebook.snapshot()
        update = state.notebook.apply_turn(result.text, truncated=output.finish_reason == 'length')
        state.memory_errors = list(dict.fromkeys(update.errors))[:8]
        metadata = dict(harness='numbered_notebook', control=update.control, events=update.events,
            errors=update.errors, incomplete_action=update.incomplete_action,
            memory_before=before, memory_after=state.notebook.snapshot())
        prompt=result.prompt
        if isinstance(prompt,Mapping):
            prompt=prompt.get('prompt','')
        state.turns.append(build_trajectory_turn(turn_index=len(state.turns)+1, prompt=prompt,
            response=result.text, result=result, metadata=metadata))
        if update.control == 'answer':
            state.finished = True
            state.final_response = update.response if not update.errors else '[Invalid memory/answer protocol]'
            state.terminal_reason = 'answer_submitted' if not update.errors else 'protocol_error'
        elif len(state.turns) >= self.max_turns:
            self.force_finish(state, 'max_turns')
        elif update.control not in ('continue', 'length'):
            self.force_finish(state, 'missing_control')
        state.metadata.update(memory=state.notebook.snapshot(), memory_errors=state.memory_errors)

    def force_finish(self, state, reason):
        state.finished = True
        state.terminal_reason = reason
        state.final_response = '[No final answer]'
