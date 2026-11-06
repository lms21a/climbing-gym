from dataclasses import replace
import json
from types import SimpleNamespace

import pytest

from climbing_gym.config import Inference, Rollout
from climbing_gym.inference.config import InferenceEngineConfig, SamplingConfig
from climbing_gym.inference.lfm25_async_engine import LFM25AsyncInferenceEngine
from climbing_gym.preflight import resolve_limits, validate_preflight


def configuration(tmp_path, **inference_options):
    inference = Inference(model_path=str(tmp_path / "model"), compile_decode=False,
                          preflight_report_path=str(tmp_path / "speed.json"), **inference_options)
    rollout = Rollout(batch_size=4, effective_batch_size=4, num_rollouts=2,
                      sampling=SamplingConfig(max_tokens=4096, logprobs=1))
    return SimpleNamespace(inference=inference, rollout=rollout)


def report(config, *, speed=1200):
    return {"engine": "native_lfm25", "config": {"model": config.inference.model_path,
            "output_tokens": 4096, "prompts_json": "/representative.json", "device": "cuda", "dtype": "auto"},
            "sampling": {"return_logprobs": True}, "max_model_len": 8192,
            "batches": [{"batch_size": 8, "median_tokens_per_second": speed,
                         "peak_allocated_bytes": 1_000_000, "peak_reserved_bytes": 2_000_000,
                         "runs": [{"seconds": 10, "output_tokens": 8 * 4096, "tokens_per_second": speed}]}]}


def save(config, value):
    from pathlib import Path
    Path(config.inference.preflight_report_path).write_text(json.dumps(value))


def test_preflight_initial_baseline_and_all_resolved_limits(tmp_path):
    config = configuration(tmp_path)
    save(config, report(config))
    result = validate_preflight(config)
    assert result["limits"]["native_batch_size"] == 64
    assert result["limits"]["max_inflight_requests"] == 8
    assert result["limits"]["representative_batch_size"] == 8
    assert result["comparison"] == {"status": "initial_baseline", "prior_report_path": None}


def test_preflight_prior_comparison_retains_slowdown_without_runtime_guard(tmp_path):
    prior_path = tmp_path / "prior.json"
    config = configuration(tmp_path, preflight_prior_report_path=str(prior_path))
    prior_path.write_text(json.dumps(report(config, speed=1000)))
    save(config, report(config, speed=100))
    result = validate_preflight(config)
    assert result["comparison"]["tokens_per_second_ratio"] == .1
    assert result["comparison"]["peak_allocated_bytes_ratio"] == 1


@pytest.mark.parametrize("mutation,match", [
    (lambda r: r["config"].update(output_tokens=256), "token budget"),
    (lambda r: r["config"].update(model="/other"), "checkpoint"),
    (lambda r: r["config"].update(prompts_json=None), "representative"),
    (lambda r: r["sampling"].update(return_logprobs=False), "log probabilities"),
    (lambda r: r["batches"][0].update(batch_size=1), "batch size 8"),
    (lambda r: r["batches"][0].update(peak_allocated_bytes=0), "peak_allocated"),
    (lambda r: r["batches"][0]["runs"][0].update(output_tokens=7), "complete"),
    (lambda r: r["config"].update(cuda_graphs=True), "cuda_graphs"),
])
def test_preflight_rejects_unrepresentative_artifacts(tmp_path, mutation, match):
    config = configuration(tmp_path)
    value = report(config)
    mutation(value)
    save(config, value)
    with pytest.raises(ValueError, match=match):
        validate_preflight(config)


def test_limits_resolve_minimum_and_reject_invalid_values(tmp_path):
    config = configuration(tmp_path, max_num_seqs=128, native_max_batch_size=64)
    rollout = replace(config.rollout, max_inflight_requests=32)
    assert resolve_limits(config.inference, rollout)["representative_batch_size"] == 32
    invalid = SimpleNamespace(**{**vars(SimpleNamespace(**{name: getattr(rollout, name) for name in (
        "batch_size", "effective_batch_size", "num_rollouts", "max_inflight_requests", "max_total_requests")})),
        "max_inflight_requests": 0})
    with pytest.raises(ValueError, match="max_inflight_requests"):
        resolve_limits(config.inference, invalid)


def test_missing_artifact_has_actionable_command(tmp_path, monkeypatch):
    config = configuration(tmp_path)
    config.inference = replace(config.inference, preflight_report_path="")
    monkeypatch.delenv("CLIMBING_GYM_PREFLIGHT_REPORT", raising=False)
    with pytest.raises(ValueError, match="climbing_gym.inference.benchmark"):
        validate_preflight(config)


@pytest.mark.parametrize("apply_template,prompt,thinking,prefill", [
    (False, "plain request", True, ""),
    (True, "thinking request", True, "<think>"),
    (True, "chat request", False, ""),
    (True, {"prompt": "already rendered", "prompt_token_ids": [12, 13]}, True, "<think>"),
    (True, [8, 9], False, "prefix"),
])
def test_tokenizer_only_prompt_renderer_matches_native_adapter(apply_template, prompt, thinking, prefill):
    class Tokenizer:
        def encode(self, text, add_special_tokens=False):
            return list(map(ord, text))
        def decode(self, ids, skip_special_tokens=False):
            return "".join(map(chr, ids))
        def apply_chat_template(self, messages, *, enable_thinking, **kwargs):
            return f"<user>{messages[0]['content']}</user><assistant thinking={enable_thinking}>"
    tokenizer = Tokenizer()
    cfg = InferenceEngineConfig(model_path="unused", apply_chat_template=apply_template,
                                chat_template_enable_thinking=thinking, assistant_prefill=prefill)
    # A CPU stub owns no model and cannot generate. Both entry points must use
    # identical rendering, including exact token-ID requests and prefills.
    engine = LFM25AsyncInferenceEngine.from_components(engine=object(), tokenizer=tokenizer, config=cfg)
    expected = engine.tokenize_prompt(prompt)
    actual = LFM25AsyncInferenceEngine.tokenize_for_config(cfg, tokenizer, prompt)
    assert expected == actual


def test_graph_bf16_packing_defaults_must_match(tmp_path):
    config = configuration(tmp_path, native_cuda_graphs=True)
    value = report(config)
    value["config"]["cuda_graphs"] = True
    save(config, value)
    with pytest.raises(ValueError, match="bf16-weight-packing"):
        validate_preflight(config)
    value["config"]["bf16_weight_packing"] = True
    save(config, value)
    assert validate_preflight(config)["tokens_per_second"] > 0


def test_prepare_real_requests_uses_tokenizer_only_and_eval_profile(tmp_path, monkeypatch):
    from climbing_gym.preflight import prepare_prompts
    from climbing_gym.config import Data, Env, Harness
    rows = [{"question": "How many?", "answer": "#### 2"}, {"question": "How much?", "answer": "#### 4"}]
    config = SimpleNamespace(data=SimpleNamespace(load_rows=lambda: rows, seed=3, shuffle=True),
                             env=Env(name="gsm8k"), harness=Harness(), inference=configuration(tmp_path).inference,
                             rollout=configuration(tmp_path).rollout,
                             eval=SimpleNamespace(seed=1, shuffle=False, max_examples=1))
    manager = SimpleNamespace(config=config, config_path=tmp_path / "job.toml")
    monkeypatch.setattr("climbing_gym.config.manager.ConfigManager.from_toml", lambda path: manager)
    class Tokenizer:
        def encode(self, text, **kwargs):
            return list(map(ord, text))
        def decode(self, ids, **kwargs):
            return "".join(map(chr, ids))
    monkeypatch.setattr("climbing_gym.models.lfm2_compat.load_lfm2_tokenizer", lambda path: Tokenizer())
    result = prepare_prompts(manager.config_path, tmp_path / "prompts.json", evaluation=True)
    assert len(result["prompts"]) == 1
    assert "How many?" in result["examples"][0]["prompt"]
    assert result["limits"]["representative_batch_size"] == 2
    assert result["examples"][0]["expected_answer"] == "2"


def test_periodic_eval_can_use_separate_budget_and_report(tmp_path):
    from climbing_gym.preflight import validate_eval_preflight
    config = configuration(tmp_path)
    alternate = tmp_path / "eval-speed.json"
    config.eval = SimpleNamespace(preflight_report_path=str(alternate), preflight_prior_report_path="")
    value = report(config)
    value["config"]["output_tokens"] = 128
    value["batches"][0]["batch_size"] = 2
    value["batches"][0]["runs"][0]["output_tokens"] = 256
    alternate.write_text(json.dumps(value))
    coordinator = Rollout(batch_size=2, effective_batch_size=2, num_rollouts=1,
                          sampling=SamplingConfig(max_tokens=128))
    run = SimpleNamespace(rollout_coordinator=SimpleNamespace(config=coordinator.build_coordinator_config()))
    result = validate_eval_preflight(config, run)
    assert result["max_tokens"] == 128
    assert result["limits"]["representative_batch_size"] == 2
