# pyright: strict
import subprocess

def format_python(content: str) -> str:
    return subprocess.run(["black", "--code", content], capture_output=True).stdout.decode()
