"""API authentication-token bootstrapping.

Resolution order (highest priority first):
1. `API_AUTH_TOKEN` environment variable (operator override).
2. `<DATA_DIR>/.api_token` file (auto-generated on first run).
3. Generate fresh `secrets.token_hex(32)`, persist with mode 0600, return it.

The token file is mode 0600 so it isn't readable by other UIDs on the host.
The web-ui container reads the same file via a read-only volume mount, so the
proxy in the Next.js server attaches the bearer token server-side without the
value ever appearing in the client JS bundle.
"""
import os
import secrets

import settings


_TOKEN_FILENAME = ".api_token"


def _token_path() -> str:
    return os.path.join(settings.DATA_DIR, _TOKEN_FILENAME)


def ensure_api_token() -> str:
    """Return the API auth token, generating and persisting one if needed."""
    if settings.API_AUTH_TOKEN:
        return settings.API_AUTH_TOKEN

    path = _token_path()

    # EAFP read: avoids a TOCTOU between an existence check and the open.
    try:
        with open(path, "r", encoding="utf-8") as f:
            existing = f.read().strip()
        if existing:
            return existing
    except FileNotFoundError:
        pass

    os.makedirs(os.path.dirname(path), exist_ok=True)
    token = secrets.token_hex(32)

    # Atomic-claim: O_EXCL so two processes racing on first start don't both
    # write. Loser falls back to reading whatever the winner persisted.
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(token + "\n")
        return token
    except FileExistsError:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
