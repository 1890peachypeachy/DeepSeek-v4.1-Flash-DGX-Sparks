"""CPU check for adapter/autotune_keep.py: per-rank caches with disjoint shape entries digest alike
when every rank's sidecar matches its launch, and differently (-> dropped) when one does not."""
import contextlib
import json
import os
import tempfile
import types
from pathlib import Path

os.environ["DSV41_AUTOTUNE_KEEP"] = "1"
import autotune_keep as ak  # noqa: E402


def fake_module(paths, contents):
    mod = types.SimpleNamespace()
    mod._autotune_cache_digest = lambda p, env: "stock"
    seen = {}

    @contextlib.contextmanager
    def ctx(model_runner, **kw):
        seen[model_runner] = paths[model_runner].is_file()   # what tuning started from
        yield
        paths[model_runner].write_text(contents[model_runner])   # tuning saves its cache

    mod.seen = seen
    mod.flashinfer_autotune_context = ctx
    mod.flashinfer_autotune_cache_path = lambda mr: paths[mr]
    return mod


def scenario(tp4):
    if tp4:
        os.environ["DSV41_LAUNCHER"] = "tp4"
    else:
        os.environ.pop("DSV41_LAUNCHER", None)
    for key in ("SGLANG_RUN_ID", "DSV41_REPLICATED_SPLIT", "DSV41_WO_A_W8_DRAFT",
                "DSV41_VERIFY_CAP", "SGLANG_SOMETHING_SHAPED"):
        os.environ.pop(key, None)
    d = Path(tempfile.mkdtemp())
    meta = {"flashinfer": "0.6.18", "arch": "sm121"}
    paths = {}
    contents = {}
    for rank, shapes in ((0, ["(192, 2304, 320)"]), (1, ["(64, 2304, 320)"])):
        p = d / f"rank_tp{rank}_pp0_dp0.json"
        contents[rank] = json.dumps({"_metadata": meta, **{s: [16, 36] for s in shapes}})
        p.write_text(contents[rank])
        paths[rank] = p
    mod = fake_module(paths, contents)
    ak.install(mod)
    env = {"cuda": "13.0"}
    assert mod._autotune_cache_digest(paths[0], env) == "", "no sidecar yet: must not be kept"
    if tp4:
        assert not paths[0].exists(), "tp4 digest deletes a cache without a matching sidecar"
    else:
        assert paths[0].exists(), "start.sh digest does not delete; the context does"
    for rank in (0, 1):                                  # a tuning boot writes the caches and sidecars
        with mod.flashinfer_autotune_context(rank):
            pass
    assert mod.seen == {0: False, 1: False}, "a cache without this launch's sidecar must not be loaded"
    for rank in (0, 1):                                  # identical relaunch: kept and loaded
        with mod.flashinfer_autotune_context(rank):
            pass
    assert mod.seen == {0: True, 1: True}, "a cache with a matching sidecar must be kept"
    d0, d1 = mod._autotune_cache_digest(paths[0], env), mod._autotune_cache_digest(paths[1], env)
    assert d0 and d0 == d1, "disjoint EP shape sets with matching launches must digest alike"
    ak.sidecar(paths[1]).write_text("other-launch\n")
    assert mod._autotune_cache_digest(paths[1], env) == "", "a changed launch must drop the cache"
    if tp4:
        assert not paths[1].exists(), "and delete it, so no rank can load it"
    else:
        assert paths[1].exists(), "start.sh leaves the file for the context to drop"
    fp_before = ak.launch_fingerprint()
    os.environ["SGLANG_RUN_ID"] = "sglang-run-1.0-1"   # per boot, set by the engine
    assert ak.launch_fingerprint() == fp_before
    os.environ["SGLANG_RUN_ID"] = "sglang-run-2.0-2"
    assert ak.launch_fingerprint() == fp_before
    os.environ["DSV41_REPLICATED_SPLIT"] = "wqkv_a"
    if tp4:
        assert ak.launch_fingerprint() == fp_before, "decode-side switch is volatile on the TP4 launcher"
    else:
        assert ak.launch_fingerprint() != fp_before, "start.sh still fingerprints this switch"
    os.environ.pop("DSV41_REPLICATED_SPLIT")
    fp_before = ak.launch_fingerprint()
    os.environ["DSV41_WO_A_W8_DRAFT"] = "1"              # draft einsum route: volatile on both paths
    assert ak.launch_fingerprint() == fp_before
    os.environ["DSV41_VERIFY_CAP"] = "conf:0.1"          # A/B-neutral switch: same fingerprint
    assert ak.launch_fingerprint() == ak.launch_fingerprint()
    fp_a = ak.launch_fingerprint()
    os.environ["DSV41_VERIFY_CAP"] = "3"
    assert ak.launch_fingerprint() == fp_a
    os.environ["SGLANG_SOMETHING_SHAPED"] = "1"
    assert ak.launch_fingerprint() != fp_a


def main():
    scenario(False)
    scenario(True)
    print("test_autotune_keep: ok")


if __name__ == "__main__":
    main()
