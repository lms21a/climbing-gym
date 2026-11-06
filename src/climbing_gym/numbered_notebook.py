"""Literal, bounded, per-trajectory notes with automatically assigned numbers.

This module interprets only the memory protocol. It never interprets, verifies,
or repairs the contents of a note.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Callable


PREFILL = '<thinking>\nI will decompose the problem into smaller deductions and solve each one carefully.\n'
_OPEN = re.compile(r'<(?:save|edit|delete|continue|answer)\b[^>]*>')
_RESERVED = re.compile(r'</?(?:save|edit|delete|continue|answer|thinking)\b')


@dataclass
class TurnUpdate:
    response: str
    events: list[dict] = field(default_factory=list)
    control: str = ''
    errors: list[str] = field(default_factory=list)
    incomplete_action: bool = False


class NumberedNotebook:
    def __init__(self, encode: Callable, *, max_tokens: int = 1024, max_entries: int = 32):
        if max_tokens <= 0 or max_entries <= 0:
            raise ValueError('Notebook limits must be positive')
        self.encode = encode
        self.max_tokens = int(max_tokens)
        self.max_entries = int(max_entries)
        self.entries: dict[int, str] = {}
        self.next_number = 1

    def render(self, entries=None):
        entries = self.entries if entries is None else entries
        return '\n'.join(f'[{number}] {text}' for number, text in sorted(entries.items())) or '(empty)'

    @property
    def token_count(self):
        return len(self.encode(self.render()))

    def snapshot(self):
        return dict(entries={str(k): v for k, v in self.entries.items()},
                    next_number=self.next_number, tokens=self.token_count)

    def apply(self, command: str, text: str = '', number: int | None = None):
        candidate = dict(self.entries)
        if command in ('save', 'edit'):
            if not text.strip() or _RESERVED.search(text):
                raise ValueError('Note must be nonempty literal text without reserved control tags')
        if command == 'save':
            if number is not None:
                raise ValueError('Save numbers are assigned by the harness')
            number = self.next_number
            candidate[number] = text
        elif command in ('edit', 'delete'):
            if type(number) is not int or number not in candidate:
                raise ValueError('Note number does not exist')
            if command == 'edit':
                candidate[number] = text
            else:
                del candidate[number]
        else:
            raise ValueError('Unknown memory action')
        if len(candidate) > self.max_entries:
            raise ValueError('Notebook entry limit exceeded')
        if len(self.encode(self.render(candidate))) > self.max_tokens:
            raise ValueError('Notebook token limit exceeded')
        self.entries = candidate
        if command == 'save':
            self.next_number += 1
        return dict(command=command, number=number, ok=True, tokens=self.token_count)

    def apply_turn(self, text: str, *, truncated: bool = False) -> TurnUpdate:
        """Apply complete operations in order, preserving earlier valid writes.

        The generated thinking opener is supplied by the prompt. Memory actions
        may appear during reasoning. Only the final answer block is terminal.
        An incomplete last action at a length cutoff is ignored, never applied.
        """
        text = str(text).removesuffix('<|im_end|>').removesuffix('<|endoftext|>')
        result = TurnUpdate(response='')
        kept, cursor = [], 0
        while match := _OPEN.search(text, cursor):
            kept.append(text[cursor:match.start()])
            opener = match.group()
            if opener == '<continue>':
                result.control = 'continue'
                if text[match.end():].strip():
                    result.errors.append('Text after continue')
                cursor = len(text)
                break
            if opener == '<answer>':
                ending = text.find('</answer>', match.end())
                if ending < 0:
                    result.incomplete_action = True
                    if not truncated:
                        result.errors.append('Incomplete answer')
                    cursor = len(text)
                    break
                tail = text[ending + len('</answer>'):]
                if tail.strip() or _RESERVED.search(text[match.end():ending]):
                    result.errors.append('Malformed terminal answer')
                kept.append(text[match.start():ending + len('</answer>')])
                result.control = 'answer'
                cursor = len(text)
                break
            command = opener[1:].split()[0].rstrip('>')
            close = f'</{command}>'
            ending = text.find(close, match.end())
            if ending < 0:
                result.incomplete_action = True
                if not truncated:
                    result.errors.append('Incomplete memory action')
                cursor = len(text)
                break
            body = text[match.end():ending]
            header = re.fullmatch(r'<(save|edit|delete)(?: ([1-9][0-9]*))?>', opener)
            number = None
            try:
                number = int(header[2]) if header and header[2] else None
                if not header or (command == 'save' and number is not None) or (
                    command in ('edit', 'delete') and number is None
                ) or (command == 'delete' and body.strip()):
                    raise ValueError('Malformed memory action')
                event = self.apply(command, body, number)
            except ValueError as error:
                event = dict(command=command, number=number, ok=False, error=str(error))
                result.errors.append(str(error))
            result.events.append(event)
            cursor = ending + len(close)
        kept.append(text[cursor:])
        result.response = ''.join(kept)
        # Catch unmatched closers, malformed openers, and nested protocol text.
        residual = re.sub(r'<answer>.*?</answer>', '', result.response, flags=re.S)
        if re.search(r'</?(?:save|edit|delete|continue|answer)\b', residual):
            result.errors.append('Unmatched control tag')
        if not result.control:
            result.control = 'length' if truncated else 'missing_control'
        return result


def notebook_instruction(*, turn: int, max_turns: int, turn_tokens: int, notebook: NumberedNotebook,
                         feedback: list[str] = ()) -> str:
    text = (
        'Work on the task using a persistent notebook. You control its contents.\n'
        f'Turn {turn} of {max_turns}; at most {turn_tokens} generated tokens, including all memory actions.\n'
        f'The notebook may contain at most {notebook.max_tokens} tokens and {notebook.max_entries} entries.\n'
        'When you are ready to record progress, close </thinking>, then issue memory actions:\n'
        'To save: open <save>, write your actual progress, then close </save>. A number is assigned automatically.\n'
        'To edit an existing note: use its numbered edit opener, write the complete new content, then close </edit>.\n'
        'To delete: use a numbered delete tag followed immediately by </delete>, with no text between them.\n'
        'Other numbers do not change after deletion; numbers are never reused.\n'
        'Use the actual existing note number for edit/delete. Multiple actions are applied in written order.\n'
        'The next turn receives the original task and all remaining notes. Other scratchwork is discarded.\n'
        'Save useful unfinished work before continuing. Notes can contain tentative ideas, corrections, '
        'partial results, or next steps; you are responsible for their accuracy.\n'
        'To finish, close </thinking> and put your actual final answer between <answer> and </answer>.\n'
        f'Next save number: {notebook.next_number}.\n'
    )
    if notebook.entries:
        ids=sorted(notebook.entries)
        text += 'Existing edit openers: '+', '.join(f'<edit {i}>' for i in ids)+'.\n'
        text += 'Existing delete actions: '+', '.join(f'<delete {i}></delete>' for i in ids)+'.\n'
        text += 'Choose the number of the note you actually intend to change or remove.\n'
    else:
        text += 'The notebook is empty. Save a note before trying to edit or delete it.\n'
    if turn == max_turns:
        text += 'This is the final turn. No further turn is available. Submit your final answer after any memory updates.\n'
    else:
        text += 'After saving progress, end with <continue> to request the next turn.\n'
    if feedback:
        text += 'Previous memory errors: ' + '; '.join(feedback) + '\n'
    return text


def render_notebook_prompt(tokenizer, task: str, notebook: NumberedNotebook, *, turn: int,
                           max_turns: int = 4, turn_tokens: int = 4096, feedback=()) -> str:
    content = (notebook_instruction(turn=turn, max_turns=max_turns, turn_tokens=turn_tokens,
                                   notebook=notebook, feedback=feedback)
               + '\nOriginal task:\n' + task + '\n\nCurrent notebook:\n' + notebook.render())
    return tokenizer.apply_chat_template([dict(role='user', content=content)], tokenize=False,
                                        add_generation_prompt=True, enable_thinking=False) + PREFILL
