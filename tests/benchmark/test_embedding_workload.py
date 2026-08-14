"""Embedding-recipe bench command and smoke-response checks."""

from emmy.benchmark.workload import build_bench_command
from emmy.deploy.orchestrate import _check_chat_response, _check_completion_response, _check_embedding_response, _smoke_response_check
from emmy.recipe.types import Recipe


def _recipe(task: str) -> Recipe:
    return Recipe.from_dict(
        {
            "model": {"huggingface": "Qwen/Qwen3-Embedding-0.6B", "task": task},
            "engine": {"llm": {"context_length": 4096, "vllm": {}}},
            "benchmark": {"max_concurrency": 8, "num_prompts": 32, "random_input_len": 128, "random_output_len": 1},
        }
    )


def test_embed_bench_command_targets_embeddings_endpoint():
    cmd = build_bench_command(_recipe("embed"))
    assert "--backend openai-embeddings" in cmd
    assert "--endpoint /v1/embeddings" in cmd
    assert "--random-output-len" not in cmd
    assert "--random-input-len 128" in cmd


def test_generate_bench_command_unchanged():
    cmd = build_bench_command(_recipe("generate"))
    assert "--backend" not in cmd
    assert "--random-output-len 1" in cmd


def test_check_embedding_response():
    good = '{"data": [{"embedding": [0.6, 0.8], "index": 0}]}'
    assert _check_embedding_response(good)[0] == "pass"
    nan = '{"data": [{"embedding": [1.0, null], "index": 0}]}'
    assert _check_embedding_response(nan)[0] in ("fail", "retry")
    unnormalized = '{"data": [{"embedding": [3.0, 4.0], "index": 0}]}'
    verdict, detail = _check_embedding_response(unnormalized)
    assert verdict == "fail" and "norm" in detail
    not_ready = '{"error": "loading"}'
    assert _check_embedding_response(not_ready)[0] == "retry"
    assert _check_embedding_response("not json")[0] == "retry"


def test_check_chat_response():
    assert _check_chat_response('{"choices": [{"message": {"content": "The answer is 4."}}]}')[0] == "pass"
    assert _check_chat_response('{"choices": [{"message": {"content": "five"}}]}')[0] == "fail"
    assert _check_chat_response("oops")[0] == "retry"
    assert _check_chat_response('{"choices": [{"message": {}}]}')[0] == "retry"


def test_check_completion_response():
    assert _check_completion_response('{"choices": [{"text": " 4\\n"}]}')[0] == "pass"
    assert _check_completion_response('{"choices": [{"text": " five"}]}')[0] == "fail"
    assert _check_completion_response("oops")[0] == "retry"
    assert _check_completion_response('{"choices": [{}]}')[0] == "retry"


def test_bench_readiness_does_not_judge_model_output():
    cases = [
        (_recipe("generate"), '{"choices": [{"message": {"content": "five"}}]}'),
        (
            Recipe.from_dict(
                {
                    "model": {"huggingface": "org/base", "task": "generate", "smoke_test": "completion"},
                    "engine": {"llm": {"vllm": {}}},
                }
            ),
            '{"choices": [{"text": "five"}]}',
        ),
        (_recipe("embed"), '{"data": [{"embedding": [3.0, 4.0]}]}'),
    ]
    for recipe, response in cases:
        check = _smoke_response_check(recipe, check_smoke_output=False)
        assert check(response)[0] == "pass"
