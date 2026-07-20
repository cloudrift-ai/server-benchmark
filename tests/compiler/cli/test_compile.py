"""CLI tests for ``emmy compile`` argument handling and ``--code``."""


def test_compile_code_torch_ir(run_cli):
    rc, stdout, stderr = run_cli("compile", "--code", "torch.nn.RMSNorm(2048)(torch.randn(1,32,2048))", "--ir", "torch")
    assert rc == 0, f"stderr: {stderr}"
    assert "rms_norm" in stdout
    assert "(1, 32, 2048)" in stdout


def test_compile_code_tensor_ir(run_cli):
    rc, stdout, stderr = run_cli("compile", "--code", "torch.nn.Linear(3, 2, bias=False)(torch.randn(4, 3))", "--ir", "tensor")
    assert rc == 0, f"stderr: {stderr}"
    assert "multiply(" in stdout
    assert "sum(" in stdout


def test_compile_code_loop_ir(run_cli):
    rc, stdout, stderr = run_cli("compile", "--code", "torch.nn.ReLU()(torch.randn(8))", "--ir", "loop")
    assert rc == 0, f"stderr: {stderr}"
    assert "===" in stdout
    assert "load " in stdout
    assert " = relu(" in stdout


def test_compile_unfused_softmax_loopify_survives_dim(run_cli, monkeypatch):
    """Regression: the readability re-roller (``100_loopify``) crashed reconstructing a ``Dim``.

    Cutting flash into a score producer + a two-pass softmax/P@V consumer
    (``PLACE@fold=cut,PLACE@tuple=cut``) produces a re-rollable statement run whose divergence bottoms
    out in a ``Dim`` — a ``dataclass(init=False)`` whose ``__init__(value, …)`` param name differs from
    its ``expr`` field. ``_reroll``'s generic ``type(v0)(**{field: …})`` reconstruction then raised
    ``TypeError: Dim.__init__() got an unexpected keyword argument 'expr'``. ``compile`` renders with
    the re-roller ON by default, so this is the exact user-facing repro. The knob rides ``EMMY_KNOBS``
    (inherited by the ``run_cli`` subprocess); the small listing shape reproduces it."""
    sdpa = (
        "q=torch.randn(1,4,128,64,dtype=torch.float16);"
        "k=torch.randn(1,4,128,64,dtype=torch.float16);"
        "v=torch.randn(1,4,128,64,dtype=torch.float16);"
        "F.scaled_dot_product_attention(q,k,v)"
    )
    monkeypatch.setenv("EMMY_KNOBS", "PLACE@fold=cut,PLACE@tuple=cut")
    rc, stdout, stderr = run_cli("compile", "--code", sdpa, "--ir", "cuda")
    assert rc == 0, f"loopify crashed on the un-fused softmax:\n{stderr}"
    assert "__global__" in stdout


def test_compile_code_saves_default_cuda_to_output(run_cli, tmp_path):
    """Default ``handle_compile`` path: no ``--ir`` ⇒ cuda. With ``--output``
    the rendered CUDA source lands on disk.

    Covers the full decomposition→optimization→fusion→tile→kernel→cuda
    pipeline (where ``LoopOp.__post_init__`` runs normalize_body →
    simplify_body on the first LoopOp construction).
    """
    out = tmp_path / "k.cu"
    rc, stdout, stderr = run_cli(
        "compile",
        "--code",
        "torch.nn.RMSNorm(64)(torch.randn(1, 8, 64))",
        "--output",
        str(out),
    )
    assert rc == 0, f"stderr: {stderr}"
    assert out.exists()
    text = out.read_text()
    # Default IR is cuda — expect a __global__ kernel rendered to source.
    assert "__global__" in text
    assert "k_rms_norm" in text


def test_compile_code_functional_silu(run_cli):
    """Functional expressions (non-Module callables) should trace too."""
    rc, stdout, stderr = run_cli("compile", "--code", "F.silu(torch.randn(1,32,128))", "--ir", "tensor")
    assert rc == 0, f"stderr: {stderr}"
    assert "exp(" in stdout
    assert "reciprocal(" in stdout


def test_compile_code_functional_softmax_bakes_kwargs(run_cli):
    """Non-tensor kwargs (e.g. dim=-1) must be baked, not turned into graph inputs."""
    rc, stdout, stderr = run_cli("compile", "--code", "F.softmax(torch.randn(1,32,128), dim=-1)", "--ir", "tensor")
    assert rc == 0, f"stderr: {stderr}"
    assert "1 inputs" in stdout
    assert "maximum(" in stdout and "sum(" in stdout


def test_compile_passes_shorthand(run_cli, tmp_path):
    """'dolft' should expand to decomposition/optimization/lifting/fusion/lowering/tile."""
    out = tmp_path / "out.txt"
    rc, stdout, stderr = run_cli("compile", "-c", "F.relu(torch.randn(8))", "--passes", "dolft", "-o", str(out), "-vv")
    assert rc == 0, f"stderr: {stderr}"
    log = stdout + stderr
    for name in ("frontend/decomposition", "frontend/optimization", "loop/lifting", "loop/fusion", "lowering/tile"):
        assert name in log, f"missing pass {name!r} in log"
    assert "lowering/cuda" not in log


def test_compile_dump_dir_writes_rule_application_files(run_cli, tmp_path):
    """``--dump-dir`` should produce per-rule .rules.{txt,json} files alongside per-pass graph dumps.

    Smoke check: at least one fusion rule fires on RMSNorm, the per-rule
    text snapshot has a ``=== rule ... matched at ... ===`` header, and
    the JSON sibling parses as a non-empty list.
    """
    import json

    dump = tmp_path / "dump"
    rc, _stdout, stderr = run_cli("compile", "-c", "torch.nn.RMSNorm(64)(torch.randn(1,8,64))", "--dump-dir", str(dump))
    assert rc == 0, f"stderr: {stderr}"

    # Per-pass graph dump is still produced.
    assert (dump / "01_frontend_decomposition.txt").exists()

    # Per-rule application snapshots — at least one fusion rule fires.
    rule_txts = sorted(dump.glob("*.rules.txt"))
    rule_jsons = sorted(dump.glob("*.rules.json"))
    assert rule_txts, "expected at least one .rules.txt file in the dump dir"
    assert len(rule_txts) == len(rule_jsons), "every .rules.txt should have a sibling .rules.json"

    # Inspect one of the merge_loop_ops snapshots — that rule reliably
    # fires on RMSNorm decomposition.
    fusion_txt = dump / "04_loop_fusion__010_merge_loop_ops.rules.txt"
    fusion_json = dump / "04_loop_fusion__010_merge_loop_ops.rules.json"
    assert fusion_txt.exists() and fusion_json.exists()
    text = fusion_txt.read_text()
    # Diff-style rendering with bracketing markers (see pipeline/rule_diff.py);
    # the ``f:`` prefix is the loop/fusion pass shorthand.
    assert ">>> f:010_merge_loop_ops" in text
    assert "<<< f:010_merge_loop_ops" in text
    assert "@@ matched at" in text

    records = json.loads(fusion_json.read_text())
    assert isinstance(records, list) and records
    first = records[0]
    assert "root" in first and "before" in first and "after" in first
    assert all(isinstance(n, dict) and "op_class" in n for n in first["before"])


def test_compile_no_input_errors(run_cli):
    rc, stdout, stderr = run_cli("compile")
    assert rc != 0
    assert "required" in stdout + stderr


def test_compile_code_and_input_mutually_exclusive(run_cli, tmp_path):
    dummy = tmp_path / "dummy.json"
    dummy.write_text("{}")
    rc, stdout, stderr = run_cli("compile", "--code", "x", str(dummy))
    assert rc != 0
    assert "mutually exclusive" in stdout + stderr


def test_compile_dynamic_emits_runtime_arg(run_cli):
    """``--dynamic seq_len@x:1`` traces with torch.export's dynamic_shapes
    so the rendered CUDA kernel signature gains an ``int seq_len`` arg."""
    rc, stdout, stderr = run_cli(
        "compile",
        "--code",
        "torch.nn.RMSNorm(2048)(torch.randn(1,32,2048))",
        "--dynamic",
        "seq_len@x:1",
        "--ir",
        "cuda",
    )
    assert rc == 0, f"stderr: {stderr}"
    assert "int seq_len" in stdout, f"expected ``int seq_len`` in kernel signature, got:\n{stdout[:500]}"


def test_compile_dynamic_bad_spec_rejected(run_cli):
    """Bad spec (missing ``@`` / ``:``) exits with usage error rather than
    crashing inside the tracer."""
    rc, stdout, stderr = run_cli(
        "compile",
        "--code",
        "torch.nn.RMSNorm(2048)(torch.randn(1,32,2048))",
        "--dynamic",
        "seq_len",
        "--ir",
        "cuda",
    )
    assert rc != 0
    assert "NAME@INPUT:AXIS" in stdout + stderr


def test_compile_golden_substring_resolves_dynamic(run_cli, monkeypatch):
    """``compile --golden`` accepts the same name **substring** ``eval --kernel``
    filters on, and a dynamic golden carries its recorded symbolic spec through — so
    the rendered kernel gains an ``int seq_len`` arg (the masked-tile path), proving
    the golden was resolved via compile, not just run."""
    # Hide CUDA from the subprocess: --golden scopes to the live card's recordings,
    # so the name only resolves everywhere via the off-GPU multi-card union.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    rc, stdout, stderr = run_cli("compile", "--golden", "o_proj.h4096.dynM", "--ir", "cuda")
    assert rc == 0, f"stderr: {stderr}"
    assert "int seq_len" in stdout, f"expected masked-tile ``int seq_len`` from the dynamic golden, got:\n{stdout[:500]}"


def test_compile_golden_ambiguous_substring_lists_shapes(run_cli, monkeypatch):
    """A ``--golden`` substring spanning several shapes can't select one graph — it
    exits 2 listing the matched shapes so the user can narrow it."""
    # Hide CUDA from the subprocess: the asserted shape pair only co-exists in the
    # off-GPU multi-card union, not necessarily in one card's own recordings.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    rc, stdout, stderr = run_cli("compile", "--golden", "mlp_down", "--ir", "tile")
    assert rc == 2, f"expected exit 2, got {rc}: {stdout}{stderr}"
    out = stdout + stderr
    assert "ambiguous" in out and "matmul.mlp_down.h4096" in out and "matmul.mlp_down.h4096.dynM" in out


def test_compile_golden_unknown_lists_available(run_cli):
    """An unknown ``--golden`` name exits 2 and lists the available golden names."""
    rc, stdout, stderr = run_cli("compile", "--golden", "no_such_shape", "--ir", "tile")
    assert rc == 2, f"expected exit 2, got {rc}"
    assert "unknown golden config" in stdout + stderr


def test_compile_code_fp32_stat_free_cone_demotes(run_cli):
    """An fp32 pointwise cone fused into a matmul A operand (the stat-free computed-A shape)
    has NO legal contraction row — no fp32 mma atoms — and unlike the MONOID composition no
    Map-form fork sibling: the recognizer must demote the node back to its PLANAR reduce and
    compile, not IndexError on the empty enumeration (the branch regression: this snippet
    compiled at the fork point via pre-nodification demotion)."""
    snippet = "(torch.nn.functional.silu(torch.randn(64, 256)) * torch.randn(64, 256)) @ torch.randn(256, 128)"
    rc, stdout, stderr = run_cli("compile", "--code", snippet, "--ir", "cuda")
    assert rc == 0, f"stderr: {stderr}"
    assert "__global__" in stdout, f"expected a rendered kernel, got:\n{stdout[:500]}"
