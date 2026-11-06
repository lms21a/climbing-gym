"""Stress native graph recapture across event-loop worker/stream reuse.

Ported from SimplePT memory-defaults 6edae8f diagnostic for release regression QA.
"""
import argparse
import asyncio
import faulthandler
import json
from pathlib import Path
import threading
import time

import torch

from climbing_gym.inference.lfm25_engine import LFM25Engine
from climbing_gym.models.lfm2 import LFM2Config, LFM2Model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model')
    parser.add_argument('--iterations', type=int, default=64)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.manual_seed(42)
    faulthandler.enable()
    if args.model:
        engine = LFM25Engine.from_pretrained(args.model, dtype=torch.bfloat16, max_model_len=3072)
    else:
        cfg = LFM2Config(vocab_size=257, hidden_size=128, mlp_hidden_size=192, num_hidden_layers=4,
                         num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=3072,
                         layer_types=('conv', 'full_attention', 'conv', 'full_attention'), attn_backend='eager')
        model = LFM2Model(cfg).eval()
        engine = LFM25Engine.from_model(model, device='cuda', dtype=torch.bfloat16, max_model_len=3072)
    engine.enable_cuda_graphs(max_graphs=1)
    decoder = engine._cuda_decoder
    forward = decoder._forward
    streams = []
    def traced(cache, tokens):
        streams.append(torch.cuda.current_stream().cuda_stream)
        return forward(cache, tokens)
    decoder._forward = traced
    @torch.inference_mode()
    def capture():
        entry = decoder._get_graph(256, 2176)
        entry.graph.replay()
        torch.cuda.synchronize()
        return threading.get_native_id()
    async def once():
        return await asyncio.get_running_loop().run_in_executor(None, capture)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as log:
        for iteration in range(args.iterations):
            print(json.dumps({'starting': iteration + 1}), flush=True)
            faulthandler.dump_traceback_later(30, exit=True)
            start = time.monotonic()
            worker = asyncio.run(once())
            torch.cuda.synchronize()
            decoder.clear()
            torch.cuda.empty_cache()
            faulthandler.cancel_dump_traceback_later()
            record = {'iteration': iteration + 1, 'seconds': time.monotonic() - start,
                      'worker_native_id': worker, 'capture_stream': streams[-1]}
            log.write(json.dumps(record) + '\n'); log.flush()
            print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
