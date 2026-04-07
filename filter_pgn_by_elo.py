#!/usr/bin/env python3
"""Filter PGN files by ELO rating. See scripts/filter_pgn_by_elo.py for usage."""

import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    script = Path(__file__).parent / "scripts" / "filter_pgn_by_elo.py"
    sys.exit(subprocess.call([sys.executable, str(script)] + sys.argv[1:]))
