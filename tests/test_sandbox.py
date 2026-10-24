"""Code sandbox: limits, timeouts, environment isolation and output caps (cats.sandbox)."""
from __future__ import annotations

import json
import os
import sys

import pytest

from cats.sandbox import Sandbox, SandboxLimits, run_python

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX rlimits")


def test_runs_code_and_reports_output():
    res = run_python("import sys\nprint(6 * 7)\nprint('oops', file=sys.stderr)")
    assert res.ok and res.exit_code == 0 and not res.timed_out
    assert res.stdout == "42\n" and res.stderr == "oops\n" and res.duration_ms > 0


def test_nonzero_exit_and_traceback():
    res = run_python("raise ValueError('bad input')")
    assert res.exit_code == 1 and not res.ok and "ValueError: bad input" in res.stderr


def test_wall_clock_timeout_kills_the_process_group():
    res = run_python("import subprocess, time\nsubprocess.Popen(['sleep', '30'])\ntime.sleep(30)", wall_seconds=1)
    assert res.timed_out and res.signal == "SIGKILL" and res.duration_ms < 10_000


def test_cpu_limit():
    res = run_python("while True:\n    pass", cpu_seconds=1, wall_seconds=30)
    assert not res.timed_out and res.signal == "SIGXCPU"


@pytest.mark.skipif(sys.platform == "darwin", reason="macOS does not reliably enforce RLIMIT_AS")
def test_memory_limit():
    res = run_python("x = bytearray(2 * 1024 ** 3)", memory_mb=256)
    assert res.exit_code != 0 and "MemoryError" in res.stderr


def test_file_size_limit():
    res = run_python("open('big', 'wb').write(b'0' * (4 * 2 ** 20))", max_file_mb=1)
    assert res.exit_code != 0 and "File too large" in res.stderr


def test_output_is_capped():
    res = run_python("print('x' * 1_000_000)", max_output_bytes=1000)
    assert res.ok and res.truncated and len(res.stdout) == 1000


def test_no_inherited_secrets_and_private_workdir(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-secret")
    monkeypatch.setenv("CATS_TEST_TOKEN", "hunter2")
    code = "import os, json\nprint(json.dumps({'env': dict(os.environ), 'cwd': os.getcwd()}))"
    res = run_python(code)
    out = json.loads(res.stdout)
    assert "ANTHROPIC_API_KEY" not in out["env"] and "CATS_TEST_TOKEN" not in out["env"]
    assert "hunter2" not in res.stdout and "sk-secret" not in res.stdout
    assert out["env"]["HOME"] == out["cwd"] and "cats-sbx-" in out["cwd"]
    assert not os.path.exists(out["cwd"])  # removed after the run


def test_files_stdin_and_extra_env():
    sb = Sandbox(SandboxLimits(wall_seconds=10), env={"GREETING": "hi"})
    res = sb.run("import os, helper\nprint(helper.X, input(), os.environ['GREETING'])",
                 files={"helper.py": "X = 5"}, stdin="there\n")
    assert res.stdout == "5 there hi\n"
    with pytest.raises(ValueError):
        sb.run("print(1)", files={"../escape.py": "x"})
