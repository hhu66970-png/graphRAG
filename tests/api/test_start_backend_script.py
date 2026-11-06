"""Run the PowerShell launcher against an isolated local backend fixture."""

import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys

import pytest


_BACKEND = '''
import json
import os
from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer

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
    Path('launched.pid').write_text(str(os.getpid()))
    if os.environ['LAUNCH_MODE'] == 'process_failure':
        raise SystemExit(23)
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            pid = os.getpid() if os.environ['LAUNCH_MODE'] == 'healthy' else -1
            self.wfile.write(json.dumps(dict(status='ok', pid=pid)).encode())
    HTTPServer(('127.0.0.1', int(os.environ['LAUNCH_PORT'])), Handler).serve_forever()
'''


@pytest.fixture
def launcher(tmp_path):
    shell = os.getenv("PWSH_EXECUTABLE") or shutil.which("pwsh") or shutil.which("powershell")
    if not shell:
        pytest.skip("PowerShell is required for launcher subprocess tests")
    root = tmp_path / "project with spaces"
    (root / "scripts").mkdir(parents=True)
    (root / "backend").mkdir()
    (root / "backend/__init__.py").write_text("")
    (root / "backend/api.py").write_text(_BACKEND)
    python_literal = sys.executable.replace("'", "''")
    (root / "scripts/resolve_python.ps1").write_text(f"Write-Output '{python_literal}'\n")
    source = Path(__file__).resolve().parents[2] / "start_backend.ps1"
    shutil.copyfile(source, root / source.name)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]

    def run(mode):
        env = {**os.environ, "LAUNCH_MODE": mode, "LAUNCH_PORT": str(port)}
        return subprocess.run(
            [shell, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
             "-File", str(root / "start_backend.ps1"), "-Detached",
             "-StartupTimeoutSeconds", "3"],
            cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30,
        )

    yield root, run
    pid_file = root / "launched.pid"
    if pid_file.exists():
        try:
            os.kill(int(pid_file.read_text()), signal.SIGTERM)
        except ProcessLookupError:
            pass


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
    pid = (root / "launched.pid").read_text()
    assert f"PID {pid}" in result.stdout
    assert "Backend ready" in result.stdout
