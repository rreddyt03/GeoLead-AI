import os
from pathlib import Path


def load_env_files(base_path=None):
    root = Path(base_path) if base_path else Path(__file__).resolve().parent
    env_paths = [
        root / ".env",
        root / ".env.openai",
        root / ".salesforce.env",
    ]

    for env_path in env_paths:
        if not env_path.exists():
            continue

        for line in env_path.read_text().splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            if stripped.startswith("export "):
                stripped = stripped[len("export "):].strip()

            if "=" not in stripped:
                continue

            key, value = stripped.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)