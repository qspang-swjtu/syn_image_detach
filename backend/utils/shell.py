import subprocess
from pathlib import Path
from typing import List, Optional


class CommandError(RuntimeError):
    pass


def run_command(args: List[str], cwd: Path, timeout: Optional[int] = None) -> subprocess.CompletedProcess:
    result = subprocess.run(
        args,
        cwd=str(cwd),
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip()
        stdout = result.stdout.strip()
        detail = stderr or stdout or f"Command failed with exit code {result.returncode}"
        raise CommandError(detail)
    return result
