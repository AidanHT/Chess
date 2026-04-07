#!/usr/bin/env python3
"""Decompress zst PGN files from Lichess database."""
import os
import zstandard as zstd
from pathlib import Path

def decompress_zst(zst_file: Path, output_file: Path) -> None:
    """Decompress a zst file to plain text."""
    print(f"Decompressing {zst_file.name}...")
    dctx = zstd.ZstdDecompressor()

    with open(zst_file, 'rb') as f_in:
        with open(output_file, 'wb') as f_out:
            dctx.copy_stream(f_in, f_out, write_size=2**20)  # 1 MB chunks

    size_gb = output_file.stat().st_size / 1e9
    print(f"  -> {output_file.name} ({size_gb:.2f} GB)")

if __name__ == "__main__":
    pgn_dir = Path(__file__).parent / "data" / "pgn"

    for zst_file in sorted(pgn_dir.glob("*.pgn.zst")):
        output_file = zst_file.with_suffix('')  # Remove .zst
        if output_file.exists():
            print(f"Skipping {output_file.name} (already exists)")
        else:
            decompress_zst(zst_file, output_file)
