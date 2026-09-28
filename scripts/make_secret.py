"""Create secrets/pg_password.txt with a random password (the only credential; never printed, gitignored)."""
import secrets
from pathlib import Path

path = Path(__file__).resolve().parents[1] / "secrets" / "pg_password.txt"
path.parent.mkdir(exist_ok=True)
if path.exists():
    print(f"{path} exists - left unchanged")
else:
    path.write_text(secrets.token_urlsafe(24), encoding="utf-8")
    print(f"wrote {path}")
