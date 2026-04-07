#!/usr/bin/env python3
"""Decompress zst PGN files. See scripts/decompress_pgn.py for usage."""

import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    script = Path(__file__).parent / "scripts" / "decompress_pgn.py"
    sys.exit(subprocess.call([sys.executable, str(script)] + sys.argv[1:]))
