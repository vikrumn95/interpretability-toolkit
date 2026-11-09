from interp_toolkit import CaptureSpec, active_hook_count, benchmark_capture, format_results


def test_benchmark_capture(model, tokens):
    res = benchmark_capture(model, tokens[0], {
        "none": None,
        "selective": CaptureSpec(components=["resid_pre"], layers=[0, 1]),
        "full": CaptureSpec.all(),
    }, repeats=2, warmup=1)
    by = {r.label: r for r in res}
    assert by["none"].retained_bytes == 0 and by["none"].n_tensors == 0
    assert 0 < by["selective"].retained_bytes < by["full"].retained_bytes
    assert by["selective"].retained_bytes == 2 * tokens[0].shape[1] * 768 * 4
    assert all(r.mean_ms > 0 and r.accelerator_peak_bytes is None for r in res)
    assert "selective" in format_results(res)
    assert active_hook_count(model) == 0
