"""Run the PowerShell launcher against an isolated local backend fixture."""

import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time

import pytest


_BACKEND = '''
import json
import os
from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer
import subprocess
import sys

class Driver:
    def verify_connectivity(self):
        if os.environ['LAUNCH_MODE'] == 'probe_failure':
            raise ConnectionError('test database unavailable')
    def close(self):
        pass
driver = Driver()

def _server_options():
    return dict(host='127.0.0.1', port=int(os.environ['LAUNCH_PORT']), debug=False)

if __name__ == '__main__':
    Path('launched.pid').write_text(str(os.getpid()), encoding='utf-8')
    if os.environ['LAUNCH_MODE'] == 'process_failure':
        raise SystemExit(23)
    if os.environ['LAUNCH_MODE'] == 'wrong_process_tree':
        child = "import os,time; from pathlib import Path; Path('descendant.pid').write_text(str(os.getpid()), encoding='utf-8'); time.sleep(120)"
        subprocess.Popen([sys.executable, '-c', child],
                         creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            pid = os.getpid() if os.environ['LAUNCH_MODE'] == 'healthy' else int(os.environ.get('HEALTH_PID', '-1'))
            self.wfile.write(json.dumps(dict(status='ok', pid=pid)).encode())
    HTTPServer(('127.0.0.1', int(os.environ['LAUNCH_PORT'])), Handler).serve_forever()
'''


def _terminate_fixture_pid(pid):
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError as exc:
        # Windows raises WinError 87 rather than ProcessLookupError for an exited PID.
        if not isinstance(exc, ProcessLookupError) and getattr(exc, 'winerror', None) != 87:
            raise


def _pid_running(pid):
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(0x00100000, False, pid)  # SYNCHRONIZE, not termination
        if not handle:
            if ctypes.get_last_error() == 87:
                return False
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            return kernel.WaitForSingleObject(handle, 0) == 258  # WAIT_TIMEOUT
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    # Orphaned children can briefly remain as zombies under a Linux test runner.
    stat = Path(f"/proc/{pid}/stat")
    return not stat.exists() or stat.read_text(encoding="utf-8").rsplit(")", 1)[1].strip().split()[0] != "Z"


def _wait_stopped(pid):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and _pid_running(pid):
        time.sleep(0.05)
    assert not _pid_running(pid), f"Fixture process {pid} survived failed startup"


@pytest.fixture
def launcher(tmp_path):
    shell = os.getenv("PWSH_EXECUTABLE") or shutil.which("pwsh") or shutil.which("powershell")
    if not shell:
        pytest.skip("PowerShell is required for launcher subprocess tests")
    root = tmp_path / "项目 with spaces"
    (root / "scripts").mkdir(parents=True)
    (root / "backend").mkdir()
    (root / "backend/__init__.py").write_text("", encoding="utf-8")
    (root / "backend/api.py").write_text(_BACKEND, encoding="utf-8")
    python_literal = sys.executable.replace("'", "''")
    (root / "scripts/resolve_python.ps1").write_text(f"Write-Output '{python_literal}'\n", encoding="utf-8")
    source = Path(__file__).resolve().parents[2] / "start_backend.ps1"
    shutil.copyfile(source, root / source.name)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]

    def run(mode, **extra_env):
        env = {**os.environ, "LAUNCH_MODE": mode, "LAUNCH_PORT": str(port),
               "PYTHONIOENCODING": "utf-8", **extra_env}
        # A detached Windows child can inherit the launcher's capture pipe handles.
        # Capture to files so subprocess.communicate() does not wait for the healthy server.
        stdout_path, stderr_path = tmp_path / "launcher.out", tmp_path / "launcher.err"
        with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
            result = subprocess.run(
                [shell, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
                 "-File", str(root / "start_backend.ps1"), "-Detached",
                 "-StartupTimeoutSeconds", "3"],
                cwd=tmp_path, env=env, stdout=stdout, stderr=stderr, timeout=30,
            )
        result.stdout = stdout_path.read_text(encoding="utf-8-sig")
        result.stderr = stderr_path.read_text(encoding="utf-8-sig")
        return result

    yield root, run
    for name in ("descendant.pid", "launched.pid"):
        pid_file = root / name
        if pid_file.exists():
            _terminate_fixture_pid(int(pid_file.read_text(encoding="utf-8")))


def test_probe_failure_stops_before_launch(launcher):
    root, run = launcher
    result = run("probe_failure")
    assert result.returncode != 0, result.stdout + result.stderr
    assert not (root / "launched.pid").exists()
    assert "Backend ready" not in result.stdout


def test_child_failure_does_not_report_success(launcher):
    root, run = launcher
    result = run("process_failure")
    assert (root / "launched.pid").exists(), result.stdout + result.stderr
    assert result.returncode != 0
    assert "Backend ready" not in result.stdout


def test_health_from_another_process_does_not_count_as_success(launcher):
    root, run = launcher
    result = run("wrong_process")
    assert (root / "launched.pid").exists(), result.stdout + result.stderr
    assert result.returncode != 0
    assert "Backend ready" not in result.stdout


def test_healthy_child_is_reported_with_its_process_id(launcher):
    root, run = launcher
    result = run("healthy")
    assert result.returncode == 0, result.stdout + result.stderr
    pid = (root / "launched.pid").read_text(encoding="utf-8")
    assert f"PID {pid}" in result.stdout
    assert "Backend ready" in result.stdout


def test_failed_launch_stops_its_descendants(launcher):
    root, run = launcher
    result = run("wrong_process_tree")
    assert result.returncode != 0, result.stdout + result.stderr
    assert "Backend ready" not in result.stdout
    for name in ("launched.pid", "descendant.pid"):
        assert (root / name).exists(), result.stdout + result.stderr
        _wait_stopped(int((root / name).read_text(encoding="utf-8")))


def test_other_live_process_is_not_accepted_or_cleaned_up(launcher):
    _, run = launcher
    other = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    try:
        result = run("wrong_process", HEALTH_PID=str(other.pid))
        assert result.returncode != 0, result.stdout + result.stderr
        assert "Backend ready" not in result.stdout
        assert other.poll() is None, "Startup cleanup killed an unrelated process"
    finally:
        # Kill its tree too: this fixture can itself be a Windows venv redirector.
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(other.pid), "/T", "/F"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        elif other.poll() is None:
            other.terminate()
        other.wait(timeout=5)


def test_missing_windows_fixture_pid_cleanup_is_idempotent(monkeypatch):
    error = OSError("fixture process already exited")
    error.winerror = 87
    monkeypatch.setattr(os, "kill", lambda *args: (_ for _ in ()).throw(error))
    _terminate_fixture_pid(12345)


def test_fixture_cleanup_does_not_hide_permission_errors(monkeypatch):
    error = PermissionError("fixture process access denied")
    monkeypatch.setattr(os, "kill", lambda *args: (_ for _ in ()).throw(error))
    with pytest.raises(PermissionError):
        _terminate_fixture_pid(12345)
