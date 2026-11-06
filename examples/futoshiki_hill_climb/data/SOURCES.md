# Puzzle bank sources

The gzip files named by `manifest.json` contain the exact selected historical
RL puzzle rows. Uncompressed and gzip SHA256 values, original run-relative
source filenames, row counts, and selected source intervals accompany each
bank. Files shared by several stages are stored once. No model weights,
rollout log probabilities, or original source snapshots are included.

Futoshiki and Countdown puzzles were generated and independently checked by
the project's algorithmic generators/solvers. Their code is distributed with
Climbing Gym under MIT. Held-out puzzle and solution-grid exclusions are
preserved by replaying the recorded banks rather than silently regenerating
new puzzles.

The recorded GSM8K subset comes from
[OpenAI's GSM8K dataset](https://huggingface.co/datasets/openai/gsm8k).
The [upstream dataset card](https://huggingface.co/datasets/openai/gsm8k/raw/main/README.md)
declares the MIT license. Original question/answer text and `source_split`,
`source_index`, and dataset IDs are retained. GSM8K is attributed to Karl
Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser,
Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher
Hesse, and John Schulman, *Training Verifiers to Solve Math Word Problems*
(2021).

Projection/SFT corpora have a separate `projection_manifest.json` and their
own source annotations. `--data-mode regenerate` produces explicitly labeled
fresh method-replication banks; those are not byte-identical historical data.
