import types
from deploy import run_command


class DummyCompleted:
    def __init__(self, returncode=0, stdout="ok\n", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_run_command_success(monkeypatch):
    # Pretend subprocess.run succeeds
    import subprocess as sp
    monkeypatch.setattr(sp, "run", lambda *a, **k: DummyCompleted(returncode=0, stdout="yay\n"))
    ok = run_command("true", "do thing")
    assert ok is True


def test_run_command_failure(monkeypatch):
    import subprocess as sp
    monkeypatch.setattr(sp, "run", lambda *a, **k: DummyCompleted(returncode=2, stdout="", stderr="boom"))
    ok = run_command("false", "bad thing")
    assert ok is False