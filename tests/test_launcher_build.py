#!/usr/bin/env python3
"""Offline tests for the build step (`cmd_build`) of the launcher.

No Docker, SSH or network: the only thing driven here is the exclude list of the
rsync that stages the tree on the workers, because that one runs with `--delete`
against a copy of the repository and a wrong pattern removes files from it.
"""
import os
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HAVE_RSYNC = shutil.which("rsync") is not None


def build_source():
    """The body of cmd_build() from start.sh."""
    source = (ROOT / "start.sh").read_text()
    build = source[source.index("cmd_build()"):]
    return build[:build.index("\n}\n")]


def rsync_excludes(which="tp4"):
    """Excludes of one rsync arm. `tp4` is start-tp4.sh; `start` is ./start.sh."""
    build = build_source()
    begin, end = {
        "tp4": ("# rsync-tp4", "# end-rsync-tp4"),
        "start": ("# rsync-start", "# end-rsync-start"),
    }[which]
    arm = build.split(begin, 1)[1].split(end, 1)[0]
    return re.findall(r"--exclude '([^']+)'", arm)


def launcher_source():
    """start.sh from the variable defaults through the helper functions, no dispatch."""
    source = (ROOT / "start.sh").read_text()
    start = source.index("HEAD_IP=")
    end = source.index('case "$CMD" in')
    return "set -euo pipefail\n" + source[start:end]


def shell(code, directory, settings=None):
    env = {"PATH": os.environ["PATH"], "HOME": os.environ["HOME"], "ROOT": str(ROOT)}
    for name in ("MODEL_DIR", "COMMON_MODEL", "SSH_IDENTITY", "NCCL_HOST_DIR",
                 "STATE_DIR", "LOG_DIR", "WORKER_DIR", "WORKER_ENGRAM_DIR", "ENGRAM_DIR"):
        env[name] = str(directory / name)
    env.update(settings or {})
    result = subprocess.run(["bash", "-c", launcher_source() + "\n" + code],
                            env=env, cwd=ROOT, text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def check(result):
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def rsync(source, destination):
    args = ["rsync", "-aH", "--delete"]
    for pattern in rsync_excludes():
        args += ["--exclude", pattern]
    args += [f"{source}/", f"{destination}/"]
    result = subprocess.run(args, text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    return result


class BuildTargetTests(unittest.TestCase):
    """`IMAGE` is whatever the profile names, so `build` must compile that recipe."""

    def test_build_compiles_the_configured_dockerfile(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "SSH_IDENTITY.pub").write_text("ssh-ed25519 AAAA test\n")
            log = Path(tmp, "docker.log")
            shell(r"""
              BUILD_DOCKERFILE=Dockerfile.canary
              BUILD_ARGS="--build-arg PIP_INDEX=https://mirror.invalid/simple"
              WORKER_HOSTS=()
              docker() { printf '%s\n' "$*" >> "$LOG"
                         case "$1 $2" in "image inspect") echo arm64;; esac; }
              cmd_build
            """, Path(tmp), {"LOG": str(log), "DSV41_LAUNCHER": "tp4"})
            calls = [line for line in log.read_text().splitlines() if line.startswith("build ")]
            self.assertEqual(len(calls), 1, log.read_text())
            self.assertIn(f"-f {ROOT}/Dockerfile.canary", calls[0])
            self.assertIn("--build-arg PIP_INDEX=https://mirror.invalid/simple", calls[0])

    def test_worker_builds_the_same_dockerfile(self):
        build = build_source()
        self.assertIn('docker build -f "$ROOT/$dockerfile" -t "$IMAGE"', build)
        workers = build[build.index("for h in \"${WORKER_HOSTS[@]}\""):]
        self.assertIn('docker build -f $(printf \'%q\' "$dockerfile")', workers)

    def test_start_sh_builds_the_default_dockerfile(self):
        """Without DSV41_LAUNCHER, BUILD_DOCKERFILE and BUILD_ARGS are ignored."""
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "SSH_IDENTITY.pub").write_text("ssh-ed25519 AAAA test\n")
            log = Path(tmp, "docker.log")
            shell(r"""
              BUILD_DOCKERFILE=Dockerfile.canary
              BUILD_ARGS="--build-arg PIP_INDEX=https://mirror.invalid/simple"
              WORKER_HOSTS=()
              docker() { printf '%s\n' "$*" >> "$LOG"
                         case "$1 $2" in "image inspect") echo arm64;; esac; }
              cmd_build
            """, Path(tmp), {"LOG": str(log)})
            calls = [line for line in log.read_text().splitlines() if line.startswith("build ")]
            self.assertEqual(calls, [f"build -t dsv41-3x-spark:local {ROOT}"])

    def test_a_dockerfile_outside_the_repository_is_rejected(self):
        """The workers only see what the rsync staged, so an absolute path cannot work."""
        build = build_source()
        self.assertIn("BUILD_DOCKERFILE must be inside the repository", build)


@unittest.skipUnless(HAVE_RSYNC, "rsync is not installed")
class RsyncExcludesTests(unittest.TestCase):
    """cmd_build must not exclude nested directories that share a name with a root artifact."""

    def test_a_nested_models_directory_reaches_the_worker(self):
        with tempfile.TemporaryDirectory() as tmp:
            src, dst = Path(tmp, "src"), Path(tmp, "dst")
            staged = "vendor/sglang/python/sglang/srt/models"
            (src / staged).mkdir(parents=True)
            (src / staged / "deepseek_v4.py").write_text("branch")
            (dst / staged).mkdir(parents=True)
            (dst / staged / "deepseek_v4.py").write_text("stale")
            (src / "models").mkdir()
            (src / "models" / "shard.safetensors").write_text("weights")

            rsync(src, dst)

            self.assertEqual((dst / staged / "deepseek_v4.py").read_text(), "branch",
                             "a nested models/ directory never reached the worker")
            self.assertFalse((dst / "models").exists(),
                             "the checkpoint directory at the repository root was copied")

    def test_every_exclude_is_anchored(self):
        """A bare name matches any path component; the TP4 arm anchors root-level artifacts."""
        for pattern in rsync_excludes("tp4"):
            if pattern in (".env", ".env.tp4"):
                continue  # exact filenames, meant to be protected everywhere
            self.assertTrue(pattern.startswith("/"), f"unanchored exclude: {pattern}")

    def test_start_sh_keeps_its_excludes(self):
        self.assertEqual(
            rsync_excludes("start"),
            [".env", ".env.tp4", "state", "state-tp4", "logs", "logs-tp4", "models", "engram"],
        )


class ExtraContainerEnvTests(unittest.TestCase):
    """EXTRA_CONTAINER_ENV reaches the head and every worker, and a key the launcher already
    passes is replaced in place instead of repeated."""

    def args(self, settings):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            Path(d, "MODEL_DIR").mkdir()
            out = shell(r"""
              head=()
              docker_common_args head "$HEAD_IP" 3
              printf '%s\0' "${head[@]}"
              printf 'SPLIT\0'
              capture() { printf '%s\0' "$@"; }
              eval "capture $(worker_env_lines 10.0.0.2 3 1)
              end-of-command"
            """, d, settings).split("\0")
        split = out.index("SPLIT")
        pairs = lambda a: [a[i + 1] for i, x in enumerate(a[:-1]) if x == "-e"]
        return pairs(out[:split]), pairs(out[split + 1:])

    def test_start_sh_ignores_the_tp4_env_channel(self):
        head, worker = self.args({"EXTRA_CONTAINER_ENV": "DSV41_MOE_B12X_NEXT=1"})
        for env in (head, worker):
            self.assertNotIn("DSV41_LAUNCHER=tp4", env)
            self.assertFalse(any(e.startswith("DSV41_SHARED_PAD_K=") for e in env))
            self.assertNotIn("DSV41_MOE_B12X_NEXT=1", env)

    def test_unset_changes_nothing(self):
        head, worker = self.args({"DSV41_LAUNCHER": "tp4"})
        self.assertIn("DSV41_WO_A_W8=0", head)
        self.assertIn("DSV41_WO_A_W8=0", worker)
        self.assertIn("DSV41_SHARED_PAD_K=0", head)
        self.assertIn("DSV41_SHARED_PAD_K=0", worker)
        self.assertIn("DSV41_LAUNCHER=tp4", head)
        self.assertIn("DSV41_LAUNCHER=tp4", worker)

    def test_set_keys_override_once_and_new_keys_are_added(self):
        extra = "DSV41_WO_A_W8=1 DSV41_SHARED_PAD_K=1 DSV41_MOE_B12X_NEXT=1 SGLANG_DSPARK_FOLDED_SAMPLING=2"
        head, worker = self.args({"DSV41_LAUNCHER": "tp4", "EXTRA_CONTAINER_ENV": extra})
        for env in (head, worker):
            for kv in extra.split():
                self.assertEqual(sum(e.split("=", 1)[0] == kv.split("=", 1)[0] for e in env), 1, kv)
                self.assertIn(kv, env)

    def test_a_malformed_entry_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            Path(d, "MODEL_DIR").mkdir()
            with self.assertRaises(AssertionError):
                shell('head=(); docker_common_args head "$HEAD_IP" 3', d,
                      {"DSV41_LAUNCHER": "tp4", "EXTRA_CONTAINER_ENV": "NOTAPAIR"})


class CanaryDockerfileTests(unittest.TestCase):
    """Dockerfile.canary installs two wheels from PyPI; an offline site needs a mirror."""

    def test_the_pip_index_is_overridable(self):
        text = (ROOT / "Dockerfile.canary").read_text()
        self.assertIn("ARG PIP_INDEX=https://pypi.org/simple", text)
        self.assertIn('-i "$PIP_INDEX"', text)


class DockerfilePathTests(unittest.TestCase):
    """Every COPY source exists in the repository and is not excluded from the build context;
    every COPY --from=fetch source is a directory scripts/fetch_runtime.sh writes."""

    FETCHED = {"/fetched/sglang-canary/python", "/fetched/b12x/b12x", "/fetched/b12x/LICENSE",
               "/fetched/b12x_next/b12x_next", "/fetched/b12x_next/LICENSE"}

    def copies(self, name):
        text = (ROOT / name).read_text().replace("\\\n", " ")
        for line in text.splitlines():
            words = line.split()
            if words[:1] == ["COPY"]:
                yield words[1:-1]

    def test_sources_exist(self):
        ignored = [p.strip().rstrip("/") for p in (ROOT / ".dockerignore").read_text().splitlines()
                   if p.strip() and not p.startswith("#")]
        fetch = (ROOT / "scripts/fetch_runtime.sh").read_text()
        for name in ("Dockerfile", "Dockerfile.canary", "Dockerfile.canary-roce"):
            for srcs in self.copies(name):
                if srcs and srcs[0] == "--from=fetch":
                    for src in srcs[1:]:
                        self.assertIn(src, self.FETCHED, f"{name}: {src}")
                        self.assertIn(src.split("/")[2], fetch, f"{name}: {src}")
                    continue
                for src in srcs:
                    self.assertTrue((ROOT / src).exists(), f"{name}: COPY {src} does not exist")
                    self.assertNotIn(src.split("/")[0], ignored, f"{name}: {src} is in .dockerignore")

    def test_base_images_are_pinned_alike(self):
        froms = set()
        for name in ("Dockerfile", "Dockerfile.canary", "Dockerfile.canary-roce"):
            for line in (ROOT / name).read_text().splitlines():
                if line.startswith("FROM "):
                    froms.add(line.split()[1])
        self.assertEqual(len(froms), 1, froms)
        self.assertRegex(froms.pop(), r"@sha256:[0-9a-f]{64}$")


if __name__ == "__main__":
    unittest.main(verbosity=2)
