#!/usr/bin/env python3
"""
Filter PGN files by minimum player ELO rating.

Extracts games where BOTH players have ELO >= threshold.
Useful for creating high-level training datasets.
"""
import argparse
import re
from pathlib import Path
from typing import Iterator

def extract_elo(pgn_text: str) -> tuple[int | None, int | None]:
    """
    Extract WhiteElo and BlackElo from PGN headers.
    Returns (white_elo, black_elo) or (None, None) if not found.
    """
    white_match = re.search(r'\[WhiteElo "(\d+)"\]', pgn_text)
    black_match = re.search(r'\[BlackElo "(\d+)"\]', pgn_text)

    white_elo = int(white_match.group(1)) if white_match else None
    black_elo = int(black_match.group(1)) if black_match else None

    return white_elo, black_elo

def stream_pgn_games(pgn_file: Path) -> Iterator[str]:
    """
    Stream individual games from a PGN file.
    Yields complete PGN games (headers + moves).
    """
    with open(pgn_file, 'r', encoding='utf-8', errors='ignore') as f:
        game_lines = []
        for line in f:
            game_lines.append(line)
            # Games are separated by blank lines; end of game
            if line.strip() == '' and game_lines:
                game_text = ''.join(game_lines).strip()
                if game_text and '[Event' in game_text:  # Valid PGN starts with Event header
                    yield game_text
                game_lines = []
        # Don't forget last game
        if game_lines:
            game_text = ''.join(game_lines).strip()
            if game_text and '[Event' in game_text:
                yield game_text

def filter_pgn_by_elo(
    input_file: Path,
    output_file: Path,
    min_elo: int = 2000,
    verbose: bool = True
) -> tuple[int, int]:
    """
    Filter PGN file, keeping only games where both players have ELO >= min_elo.

    Returns (total_games, kept_games)
    """
    total = 0
    kept = 0

    with open(output_file, 'w', encoding='utf-8') as out_f:
        for game in stream_pgn_games(input_file):
            total += 1
            white_elo, black_elo = extract_elo(game)

            # Keep game if both players meet minimum ELO
            if white_elo and black_elo and white_elo >= min_elo and black_elo >= min_elo:
                out_f.write(game)
                out_f.write('\n\n')
                kept += 1

            if verbose and total % 10_000 == 0:
                pct = 100 * kept / total if total > 0 else 0
                print(f"  Processed {total:,} games, kept {kept:,} ({pct:.1f}%)")

    return total, kept

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Filter PGN files by minimum player ELO rating"
    )
    parser.add_argument(
        "pgn_file",
        type=Path,
        help="Input PGN file"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        help="Output PGN file (default: input_2000plus.pgn)"
    )
    parser.add_argument(
        "-e", "--min-elo",
        type=int,
        default=2000,
        help="Minimum ELO for both players (default: 2000)"
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress progress output"
    )

    args = parser.parse_args()

    if not args.pgn_file.exists():
        print(f"Error: File not found: {args.pgn_file}")
        exit(1)

    output_file = args.output or args.pgn_file.parent / f"{args.pgn_file.stem}_{args.min_elo}plus.pgn"

    print(f"Filtering {args.pgn_file.name} for ELO >= {args.min_elo}...")
    total, kept = filter_pgn_by_elo(args.pgn_file, output_file, args.min_elo, not args.quiet)

    pct = 100 * kept / total if total > 0 else 0
    print(f"\nResults:")
    print(f"  Total games: {total:,}")
    print(f"  Kept games: {kept:,} ({pct:.1f}%)")
    print(f"  Output: {output_file}")
    print(f"  Size: {output_file.stat().st_size / 1e9:.2f} GB")
