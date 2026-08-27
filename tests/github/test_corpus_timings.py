"""What a realization-corpus timing run may select, and what it may commit."""

import importlib.util
import subprocess
from pathlib import Path

import yaml

MODULE_PATH = Path(__file__).parents[2] / ".github" / "scripts" / "corpus_timings.py"
SPEC = importlib.util.spec_from_file_location("corpus_timings", MODULE_PATH)
corpus_timings = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(corpus_timings)


def _case(workspace, name, cap=(12, 0), latency=None):
    path = workspace / corpus_timings.CASES_DIR / name
    path.parent.mkdir(parents=True, exist_ok=True)
    realization = {"name": "k_matmul_abc.def", "bindings": {}, "pins": {}, "knobs": {"TILE": "f2x2"}}
    if latency is not None:
        realization["latency"] = latency
    path.write_text(
        yaml.safe_dump(
            {
                "compute_cap": list(cap),
                "programs": [{}],
                "configs": [{"program": 0, "target": {"origins": ["c"]}, "realizations": [realization]}],
            },
            sort_keys=False,
        )
    )
    return path


def _repo(workspace):
    subprocess.run(["git", "init", "-q"], cwd=workspace, check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=workspace, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=workspace, check=True)
    subprocess.run(["git", "add", "-A"], cwd=workspace, check=True)
    subprocess.run(["git", "commit", "-qm", "baseline"], cwd=workspace, check=True)


# --- selection -------------------------------------------------------------------------------


def test_selection_requires_an_exact_capability_match(tmp_path):
    """A pinned schedule is a claim about one capability, not about a merely newer card."""
    _case(tmp_path, "matmul/here.yaml", cap=(12, 0))
    _case(tmp_path, "matmul/elsewhere.yaml", cap=(8, 9))

    selected = corpus_timings.selectable(tmp_path, (12, 0))

    assert [path.name for path in selected] == ["here.yaml"]


def test_selection_skips_open_cases(tmp_path):
    """An open case's schedule never runs, so a latency for it would be a false attribution."""
    _case(tmp_path, "matmul/closed.yaml")
    _case(tmp_path, "matmul/gap_xfail_offered.yaml")

    assert [path.name for path in corpus_timings.selectable(tmp_path, (12, 0))] == ["closed.yaml"]


def test_a_card_with_no_matching_cases_is_a_clean_no_op(tmp_path):
    """A new card must be graceful rather than a failure."""
    _case(tmp_path, "matmul/here.yaml", cap=(12, 0))

    assert corpus_timings.selectable(tmp_path, (7, 0)) == []


# --- card choice -----------------------------------------------------------------------------


def _single_gpu_types():
    from emmy.hardware import GPU_INSTANCE_TYPES, resolve_instance_type

    return [
        resolve_instance_type(provider, base, 1)
        for candidates in GPU_INSTANCE_TYPES.values()
        for provider, base in candidates
        if provider == "cloudrift"
    ]


def test_card_choice_is_reproducible_from_the_run_id():
    from emmy.hardware import GPU_INSTANCE_TYPES

    available = _single_gpu_types()

    assert corpus_timings.choose_gpu(available, 12345) == corpus_timings.choose_gpu(available, 12345)
    assert corpus_timings.choose_gpu(available, 12345) in GPU_INSTANCE_TYPES


def test_card_choice_does_not_depend_on_listing_order():
    """The log records the seed; the choice must be recoverable from it alone."""
    available = _single_gpu_types()

    assert corpus_timings.choose_gpu(available, 7) == corpus_timings.choose_gpu(list(reversed(available)), 7)


def test_only_currently_rentable_cards_are_offered():
    """A card whose single-GPU instance type is not on offer must never be chosen."""
    from emmy.hardware import GPU_INSTANCE_TYPES, resolve_instance_type

    wanted = "NVIDIA GeForce RTX 5090"
    only = [resolve_instance_type("cloudrift", base, 1) for provider, base in GPU_INSTANCE_TYPES[wanted] if provider == "cloudrift"]

    assert {corpus_timings.choose_gpu(only, seed) for seed in range(20)} == {wanted}


def test_no_availability_is_reported_rather_than_guessed():
    assert corpus_timings.choose_gpu([], 7) is None
    assert corpus_timings.choose_gpu(["not-a-known-type.1"], 7) is None


# --- what a run may commit -------------------------------------------------------------------


def test_adding_a_latency_entry_is_allowed(tmp_path):
    path = _case(tmp_path, "matmul/case.yaml")
    _repo(tmp_path)
    document = yaml.safe_load(path.read_text())
    document["configs"][0]["realizations"][0]["latency"] = {"NVIDIA H100": {"emmy_us": 10.0, "tcompile_us": 12.0}}
    path.write_text(yaml.safe_dump(document, sort_keys=False))

    assert corpus_timings.validate_diff(tmp_path) == []
    assert corpus_timings.changed_files(tmp_path) == ["tests/compiler/realization/cases/matmul/case.yaml"]


def test_updating_one_card_leaves_another_cards_entry_alone(tmp_path):
    """Two runs on different cards touch disjoint lines, which is what lets one branch accumulate."""
    path = _case(tmp_path, "matmul/case.yaml", latency={"NVIDIA RTX 4090": {"emmy_us": 30.0, "tcompile_us": 24.0}})
    _repo(tmp_path)
    document = yaml.safe_load(path.read_text())
    document["configs"][0]["realizations"][0]["latency"]["NVIDIA H100"] = {"emmy_us": 10.0, "tcompile_us": 12.0}
    path.write_text(yaml.safe_dump(document, sort_keys=False))

    assert corpus_timings.validate_diff(tmp_path) == []


def test_a_moved_derived_half_is_refused(tmp_path):
    """A stale corpus must fail rather than commit — committing would fold a regeneration into a
    measurement run where nobody is reviewing it."""
    path = _case(tmp_path, "matmul/case.yaml")
    _repo(tmp_path)
    document = yaml.safe_load(path.read_text())
    document["configs"][0]["realizations"][0]["identity"] = "0" * 64
    document["configs"][0]["realizations"][0]["latency"] = {"NVIDIA H100": {"emmy_us": 10.0, "tcompile_us": 12.0}}
    path.write_text(yaml.safe_dump(document, sort_keys=False))

    assert [line.split(":")[1].strip() for line in corpus_timings.validate_diff(tmp_path)] == [
        "the derived half changed, so the corpus it measured was stale"
    ]


def test_a_changed_knob_is_refused(tmp_path):
    path = _case(tmp_path, "matmul/case.yaml")
    _repo(tmp_path)
    document = yaml.safe_load(path.read_text())
    document["configs"][0]["realizations"][0]["knobs"]["TILE"] = "f4x4"
    path.write_text(yaml.safe_dump(document, sort_keys=False))

    assert corpus_timings.validate_diff(tmp_path)


def test_a_new_case_is_refused(tmp_path):
    """Recording a gap belongs to onboarding; a timing run only measures what is already there."""
    _case(tmp_path, "matmul/case.yaml")
    _repo(tmp_path)
    _case(tmp_path, "matmul/invented.yaml")
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)

    assert any("may not add a case" in line for line in corpus_timings.validate_diff(tmp_path))


def test_untracked_output_is_refused(tmp_path):
    _case(tmp_path, "matmul/case.yaml")
    _repo(tmp_path)
    (tmp_path / "bench.log").write_text("noise\n")

    assert corpus_timings.main(["--workspace", str(tmp_path), "validate"]) == 1
