"""CPU contracts for the recorded corpus and faithful historical stage runners."""

import asyncio
from dataclasses import replace
import importlib
from pathlib import Path
import sys

import pytest
import torch

from climbing_gym.futoshiki import Puzzle, solve, render_solve_trace
from climbing_gym.trainers.packing import pack_sequences

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/futoshiki_hill_climb"
sys.path.insert(0, str(EXAMPLE))
history = importlib.import_module("historical")
native = importlib.import_module("native_projection")
workflow = importlib.import_module("projection")


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    root = tmp_path_factory.mktemp("historical_projection")
    for stage in history.PROTOCOLS:
        workflow.prepare(root, stage)
    return root


def test_recorded_corpora_counts_order_and_verified_original_contract(prepared):
    for stage, expected, grids, seeded in [
        ("solver_projection_256", 256, 195, 0),
        ("projection_reset", 2048, 2037, 21),
    ]:
        directory = workflow.stage_dir(prepared, stage)
        rows = workflow.read_rows(directory / "data/train_recorded.jsonl")
        audit = workflow.validate_records(rows, history.PROTOCOLS[stage])
        assert (
            audit["rows"],
            audit["unique_solution_grids"],
            audit["teacher_seeded"],
        ) == (expected, grids, seeded)
        guides = workflow.read_rows(directory / "data/guides_recorded.jsonl")
        assert [r["id"] for r in rows] == [r["id"] for r in guides[:expected]]
        for tier in audit["tiers"]:
            row = next(r for r in rows if r["row"]["tier"] == tier)
            batch = pack_sequences(
                [row["prompt_token_ids"] + row["completion_token_ids"]],
                prompt_lens=[len(row["prompt_token_ids"])],
                max_tokens=history.PROTOCOLS[stage].packed_tokens,
            )[0]
            mask = batch.valid_completion_mask()
            assert mask.sum() == len(row["completion_token_ids"])
            assert not mask[: len(row["prompt_token_ids"]) - 1].any()
            assert mask[-1] and batch.targets[0, -1] == 7


def test_prepare_refuses_modified_data_and_export_keeps_full_actual_traces(
    prepared, tmp_path
):
    stage = "solver_projection_256"
    workflow.prepare(tmp_path, stage)
    path = workflow.stage_dir(tmp_path, stage) / "data/train_recorded.jsonl"
    path.write_text("changed")
    with pytest.raises(ValueError, match="replace changed"):
        workflow.prepare(tmp_path, stage)
    output = tmp_path / "actual.md"
    workflow.export_samples(prepared, "projection_reset", output)
    rows = workflow.read_rows(
        workflow.stage_dir(prepared, "projection_reset") / "data/train_recorded.jsonl"
    )
    text = output.read_text()
    assert rows[0]["prompt"] in text and rows[0]["completion"] in text
    assert rows[0]["expected_answer"] in text
    assert "intermediate prose not certified" in text


def test_solver_guide_is_regenerated_exactly_from_constraints(prepared):
    guide = workflow.read_rows(
        workflow.stage_dir(prepared, "solver_projection_256")
        / "data/guides_recorded.jsonl"
    )[0]
    puzzle = Puzzle.from_row(guide["row"])
    events = []
    assert (
        solve(puzzle, trace=events)
        == solve(puzzle)
        == [tuple(guide["row"]["solution"])]
    )
    text = (
        render_solve_trace(puzzle, events)
        + "\n</thinking>\n<answer>"
        + guide["reference_answer"]
        + "</answer><|im_end|>"
    )
    assert text == guide["expert"]
    assert events[0]["event"] == "initialize" and events[-1]["event"] == "complete"


def test_historical_lr_curves_and_suffix_rules_are_distinct():
    solver = history.PROTOCOLS["solver_projection_256"]
    reset = history.PROTOCOLS["projection_reset"]
    guide = {"prompt_token_ids": [1] * 400}
    assert history.suffix_budget(solver, guide, [2] * 100, 25) == 2560 - 25
    assert history.suffix_budget(reset, guide, [2] * 100, 25) == 100 - 25
    assert history.suffix_budget(reset, guide, [], 0) == 8192
    assert history.learning_rate(solver, 0, 256) == 1e-5
    assert history.learning_rate(solver, 64, 256) == pytest.approx(5e-6)
    assert history.learning_rate(reset, 0, 2048) == pytest.approx(1e-5 / 8)
    assert history.learning_rate(reset, 8, 2048) == pytest.approx(1e-5)
    assert (solver.retained_epoch, solver.schedule_epochs) == (1, 2)
    assert (reset.retained_epoch, reset.schedule_epochs) == (2, 4)


def test_source_balanced_batch_order_and_no_score_inputs():
    rows = [
        dict(id=f"{tier}-{i}", row=dict(tier=tier))
        for i in range(20)
        for tier in history.TIERS
    ]
    first = history.training_batches(
        rows, replace(history.PROTOCOLS["projection_reset"], batch_size=8), 0
    )
    second = history.training_batches(
        rows, replace(history.PROTOCOLS["projection_reset"], batch_size=8), 0
    )
    assert first == second
    assert all(
        [r["row"]["tier"] for r in batch] == list(history.TIERS) * 4 for batch in first
    )


class Tokenizer:
    eos_token_id = 7

    def encode(self, text, add_special_tokens=False):
        pieces = text.split("<|im_end|>")
        ids = []
        for i, part in enumerate(pieces):
            if i:
                ids.append(7)
            ids.extend(ord(c) + 100 for c in part)
        return ids

    def decode(self, ids, **kwargs):
        return "".join("<|im_end|>" if i == 7 else chr(i - 100) for i in ids)

    def apply_chat_template(self, messages, **kwargs):
        return "USER:" + messages[0]["content"] + "\nASSISTANT:"


def test_specialized_prompt_keeps_first_person_and_exact_prefix(prepared):
    guide = workflow.read_rows(
        workflow.stage_dir(prepared, "projection_reset") / "data/guides_recorded.jsonl"
    )[0]
    tokenizer = Tokenizer()
    prefix = [7, 1000, 234]
    text, ids = history.proposal_prompt(tokenizer, guide, prefix)
    assert "in the first person" in text
    assert "Do not talk about the expert" in text
    assert ids == tuple(tokenizer.encode(text)) + tuple(prefix)


class Runtime:
    def __init__(self, protocol, guide, *, fail_initial=False):
        self.protocol = protocol
        self.tokenizer = Tokenizer()
        self.calls = []
        self.model_identity = dict(
            weights_sha256="frozen-student", tokenizer_sha256="same-tokenizer"
        )
        self.guide = guide
        self.fail_initial = fail_initial

    def request(self, guide, prefix, limit=None):
        return dict(prefix=prefix, limit=limit)

    async def generate(self, requests, *, seed_offset, label, **kwargs):
        self.calls.append((seed_offset, label, requests))
        outputs = []
        for request in requests:
            if self.fail_initial and not label.startswith("teacher_seed_refinement"):
                ids = self.tokenizer.encode("invalid<|im_end|>")
            else:
                ids = self.guide["expert_completion_token_ids"][
                    len(request["prefix"]) : request["limit"]
                ]
            outputs.append(
                dict(
                    completion_token_ids=ids,
                    completion_logprobs=[-3.0] * len(ids),
                    finish_reason="stop",
                )
            )
        return outputs, {}

    def score(self, guides, candidates, **kwargs):
        return [[-2.0] * len(ids) for ids in candidates]

    def emit(self, *args, **kwargs):
        pass


def fake_guide(prepared):
    source = workflow.read_rows(
        workflow.stage_dir(prepared, "projection_reset") / "data/guides_recorded.jsonl"
    )[0]
    guide = dict(source)
    guide["prompt_token_ids"] = [1, 2]
    guide["expert"] = (
        "proof\n</thinking>\n<answer>"
        + guide["reference_answer"]
        + "</answer><|im_end|>"
    )
    guide["expert_completion_token_ids"] = Tokenizer().encode(guide["expert"])
    return guide


def test_native_historical_projection_archives_resume_and_frozen_identity(
    prepared, tmp_path
):
    guide = fake_guide(prepared)
    protocol = replace(
        history.PROTOCOLS["projection_reset"],
        refinements=2,
        retries=1,
        seed_refinements=1,
    )
    runtime = Runtime(protocol, guide)
    result = asyncio.run(native.project_guides(runtime, [guide], tmp_path))
    assert result[0]["completion_token_ids"][-1] == 7
    assert result[0]["projection_attempts"] == 2
    assert not result[0]["teacher_seeded"]
    calls = len(runtime.calls)
    assert asyncio.run(native.project_guides(runtime, [guide], tmp_path)) == result
    assert len(runtime.calls) == calls
    runtime.model_identity["weights_sha256"] = "different-student"
    with pytest.raises(ValueError, match="resume"):
        asyncio.run(native.project_guides(runtime, [guide], tmp_path))


def test_teacher_seed_fallback_is_explicit_and_solver_has_no_copy_fallback(
    prepared, tmp_path
):
    guide = fake_guide(prepared)
    protocol = replace(
        history.PROTOCOLS["projection_reset"],
        refinements=1,
        retries=1,
        seed_refinements=1,
    )
    result = asyncio.run(
        native.project_guides(
            Runtime(protocol, guide, fail_initial=True), [guide], tmp_path / "reset"
        )
    )
    assert result[0]["teacher_seeded"]
    assert result[0]["teacher_origin_token_count"] == len(
        guide["expert_completion_token_ids"]
    )
    solver = replace(
        history.PROTOCOLS["solver_projection_256"], refinements=1, retries=1
    )
    with pytest.raises(RuntimeError, match="no solver-stage copy fallback"):
        asyncio.run(
            native.project_guides(
                Runtime(solver, guide, fail_initial=True), [guide], tmp_path / "solver"
            )
        )


def test_fp32_master_moments_survive_bf16_optimizer_resume():
    p = torch.nn.Parameter(torch.tensor([1.0, 2.0], dtype=torch.bfloat16))
    original = history.FP32MasterAdamW([p], lr=1e-4)
    p.grad = torch.tensor([0.1, 0.2], dtype=torch.bfloat16)
    original.step()
    q = torch.nn.Parameter(p.detach().clone())
    resumed = history.FP32MasterAdamW([q], lr=1e-4)
    import copy

    resumed.load_state_dict(copy.deepcopy(original.state_dict()))
    assert all(
        resumed.state[q][k].dtype == torch.float32
        for k in ("master", "exp_avg", "exp_avg_sq")
    )
    p.grad = torch.tensor([0.2, 0.3], dtype=torch.bfloat16)
    q.grad = p.grad.clone()
    original.step()
    resumed.step()
    torch.testing.assert_close(q, p, atol=0, rtol=0)
    torch.testing.assert_close(
        resumed.state[q]["master"], original.state[p]["master"], atol=0, rtol=0
    )


def test_sft_exact_resume_preserves_historical_schedule_and_optimizer(tmp_path):
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Embedding(16, 4)
            self.lm_head = torch.nn.Linear(4, 16)

        def get_hidden_states(self, inputs, positions):
            return self.embed(inputs)

        def save_pretrained(self, path):
            path = Path(path)
            path.mkdir(parents=True, exist_ok=True)
            torch.save(self.state_dict(), path / "weights.pt")

    class Logger:
        def __init__(self):
            self.metrics = []

        def log_step(self, **values):
            self.metrics.append(values)

    rows = [
        dict(
            id=str(i),
            row=dict(tier=history.TIERS[i % 2]),
            prompt_token_ids=[1, 2],
            completion_token_ids=[3 + i, 7],
        )
        for i in range(4)
    ]
    protocol = replace(
        history.PROTOCOLS["projection_reset"],
        batch_size=2,
        retained_epoch=2,
        head_chunk=2,
        packed_tokens=32,
    )
    torch.manual_seed(2)
    full = Tiny()
    full_optimizer = history.FP32MasterAdamW(full.parameters())
    full_logger = Logger()
    assert (
        workflow.train_loop(
            full,
            object(),
            full_optimizer,
            rows,
            protocol,
            tmp_path / "full",
            full_logger,
            device="cpu",
            evaluator=lambda *a: {},
        )
        == 4
    )
    torch.manual_seed(2)
    partial = Tiny()
    partial_optimizer = history.FP32MasterAdamW(partial.parameters())

    def interrupt(*args):
        raise RuntimeError("interrupted after saving first epoch")

    with pytest.raises(RuntimeError, match="interrupted"):
        workflow.train_loop(
            partial,
            object(),
            partial_optimizer,
            rows,
            protocol,
            tmp_path / "resume",
            Logger(),
            device="cpu",
            evaluator=interrupt,
        )
    resume = tmp_path / "resume/resume"
    state = workflow.read_json(resume / "state.json")
    assert state["pending_epoch"] == 1 and state["optimizer_updates"] == 2
    restored = Tiny()
    restored.load_state_dict(
        torch.load(resume / "policy/weights.pt", weights_only=True)
    )
    restored_optimizer = history.FP32MasterAdamW(restored.parameters())
    restored_optimizer.load_state_dict(
        torch.load(resume / "optimizer.pt", weights_only=True)
    )
    logger = Logger()
    evaluations = []
    assert (
        workflow.train_loop(
            restored,
            object(),
            restored_optimizer,
            rows,
            protocol,
            tmp_path / "resume",
            logger,
            device="cpu",
            evaluator=lambda epoch, path: evaluations.append(epoch) or {},
            resume_state=state,
        )
        == 4
    )
    assert evaluations == [1, 2]
    for key, value in full.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value, rtol=0, atol=0)
    assert [m["metrics"]["learning_rate"] for m in logger.metrics] == [
        m["metrics"]["learning_rate"] for m in full_logger.metrics[2:]
    ]


@pytest.mark.parametrize("changed", ["model_sha256", "tokenizer_sha256", "pool_sha256"])
def test_collection_cannot_relabel_stale_archives_under_new_identity(
    tmp_path, monkeypatch, changed
):
    from dataclasses import asdict

    model = tmp_path / "model"
    model.mkdir()
    (model / "pytorch_model.bin").write_bytes(b"fake fixture weights; never loaded")
    (model / "tokenizer.json").write_text("fixture vocabulary")
    directory = workflow.stage_dir(tmp_path, "projection_reset")
    data = directory / "data"
    data.mkdir(parents=True)
    (data / "calibration.jsonl").write_text("{}\n")
    tier = "5x5_g11_i5"
    pool = data / f"pool_{tier}.jsonl"
    pool.write_text("{}\n")
    identity = dict(
        model_sha256=workflow.model_digest(model),
        tokenizer_sha256=workflow.tokenizer_digest(model),
        pool_sha256=workflow.digest(pool),
        protocol=asdict(history.PROTOCOLS["projection_reset"]),
    )
    identity[changed] = "a different original identity"
    workflow.write_json(directory / f"collection/{tier}/identity.json", identity)
    monkeypatch.setattr(workflow, "validate_preflight", lambda *args: None)

    def forbidden_runtime(*args, **kwargs):
        raise AssertionError(
            "Stale collection must reject before opening native engine"
        )

    monkeypatch.setattr(workflow, "NativeRuntime", forbidden_runtime)
    with pytest.raises(ValueError, match="changed during collection resume"):
        asyncio.run(
            workflow.collect(tmp_path, model, tier=tier, report="new-preflight.json")
        )
