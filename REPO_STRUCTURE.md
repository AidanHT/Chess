# Repository Structure

```
chess/
├── chess_engine/               # Main package
│   ├── __init__.py            # Public API
│   ├── core/                  # Core types & encoding
│   │   ├── __init__.py
│   │   ├── types.py          # Constants, type aliases (from core_types.py)
│   │   └── encoding.py       # Board & move encoding (from state_encoding.py)
│   ├── data/                  # Data loading
│   │   ├── __init__.py
│   │   └── pipeline.py       # PGN streaming & DataLoader (from data_pipeline.py)
│   ├── models/                # Neural networks
│   │   ├── __init__.py
│   │   └── resnet.py         # Dual-head ResNet (from model.py)
│   ├── mcts/                  # Search algorithms
│   │   ├── __init__.py
│   │   └── search.py         # Batched MCTS (from mcts.py)
│   ├── engine/                # UCI interface
│   │   ├── __init__.py
│   │   └── uci.py            # UCI wrapper (from uci_engine.py)
│   └── training/              # Training pipelines
│       ├── __init__.py
│       ├── train.py           # Main training loop with progress tracking
│       └── high_level.py      # High-ELO focused training
├── tests/                     # Test suite
│   ├── __init__.py
│   └── smoke_test.py         # Integration tests
├── scripts/                   # Utility scripts
│   ├── decompress_pgn.py     # Decompress .zst files
│   └── filter_pgn_by_elo.py  # Filter games by player rating
├── data/                      # Data directory (in .gitignore)
│   └── pgn/                   # PGN chess games
├── checkpoints/               # Model checkpoints (in .gitignore)
│── train_engine.py            # Entry point: Standard training
├── train_high_level_engine.py # Entry point: High-ELO training
├── decompress_pgn.py          # Wrapper for scripts/decompress_pgn.py
├── filter_pgn_by_elo.py       # Wrapper for scripts/filter_pgn_by_elo.py
└── TRAINING_STRATEGY.md       # Training documentation
```

## Quick Start

### Standard Training
```bash
python train_engine.py --data_dir data/pgn --epochs 10
```

### High-Level Training (Recommended)
```bash
python train_high_level_engine.py --masters data/pgn --epochs 15
```

### Distributed Training (Multi-GPU)
```bash
torchrun --nproc_per_node=4 train_engine.py --data_dir data/pgn
```

### Utility Scripts
```bash
python decompress_pgn.py
python filter_pgn_by_elo.py data/pgn/*.pgn -e 2000
```

## Import Style

All modules are accessible via the main package:

```python
from chess_engine import ChessResNet, MCTS, UCIEngine
from chess_engine import encode_board, encode_move, decode_move
from chess_engine import make_dataloader, find_pgn_files
```

Or use submodules directly:

```python
from chess_engine.models import ChessResNet
from chess_engine.mcts import MCTS
from chess_engine.core import encode_board
from chess_engine.data import make_dataloader
from chess_engine.training import TrainConfig, train
```

## File Mapping (Old → New)

| Old Name | New Location |
|----------|------|
| `core_types.py` | `chess_engine/core/types.py` |
| `state_encoding.py` | `chess_engine/core/encoding.py` |
| `data_pipeline.py` | `chess_engine/data/pipeline.py` |
| `model.py` | `chess_engine/models/resnet.py` |
| `mcts.py` | `chess_engine/mcts/search.py` |
| `uci_engine.py` | `chess_engine/engine/uci.py` |
| `train.py` | `chess_engine/training/train.py` |
| `train_high_level.py` | `chess_engine/training/high_level.py` |
| `smoke_test.py` | `tests/smoke_test.py` |
| `decompress_pgn.py` | `scripts/decompress_pgn.py` |
| `filter_pgn_by_elo.py` | `scripts/filter_pgn_by_elo.py` |

All imports have been updated to use relative imports within the package.
