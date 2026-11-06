"""Tokenization-independent ASCII stop matching without decoding on the hot path."""
import numpy as np
import torch


def trim_after_stop(text: str, stops) -> str:
    """Keep the first completed stop string, discarding text beyond its end.

    One generated token can contain both the end of a stop and trailing text
    (for example, `></`). This changes only the displayed response: the native
    adapter retains the whole sampled token and its log probability.
    """
    ends = [start + len(stop) for stop in stops if stop and (start := text.find(stop)) >= 0]
    return text[:min(ends)] if ends else text


def build_stop_table(tokenizer, stops, *, vocab_size=None):
    if any(not s or not s.isascii() for s in stops):
        raise ValueError('Native text stops must be nonempty ASCII strings')
    prefixes = sorted({'', *(s[:i] for s in stops for i in range(1, len(s)))}, key=len)
    if len(prefixes) >= 65536:
        raise ValueError('Too many stop-prefix states')
    vocab_size = len(tokenizer) if vocab_size is None else int(vocab_size)
    if vocab_size < len(tokenizer):
        raise ValueError('Model vocabulary must cover the tokenizer vocabulary')
    pieces = [tokenizer.decode([i], skip_special_tokens=False, clean_up_tokenization_spaces=False)
              for i in range(len(tokenizer))]
    # LFM pads its output head beyond the tokenizer vocabulary. Those IDs are
    # valid samples and decode to empty text, so they preserve the prefix state.
    # Cover the model's entire output range without changing its distribution.
    pieces.extend([''] * (vocab_size - len(pieces)))
    suffixes = list(reversed(list(enumerate(prefixes))))
    table = np.zeros((len(prefixes), len(pieces)), dtype=np.int32)
    for state, prefix in enumerate(prefixes):
        for token, piece in enumerate(pieces):
            text = prefix + piece
            matched = any(stop in text for stop in stops)
            next_state = next(i for i, p in suffixes if text.endswith(p))
            table[state, token] = next_state + (65536 if matched else 0)
    return torch.from_numpy(table)


def advance_text_stops(table, states, tokens, reasons):
    value = table[states, tokens].long()
    return value & 65535, torch.where((reasons == 0) & (value >= 65536), 2, reasons)
