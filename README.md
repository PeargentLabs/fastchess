# fastchess

[![PyPI version](https://img.shields.io/pypi/v/fastchess)](https://pypi.org/project/fastchess/)
[![Python 3.9+](https://img.shields.io/pypi/pyversions/fastchess)](https://pypi.org/project/fastchess/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

A **high-performance Python chess library** written in C with Python bindings, built by [PeargentLabs](https://github.com/PeargentLabs). Designed as a **fast drop-in replacement for python-chess** in data pipelines and machine learning workflows.

**[📖 Documentation](https://peargentlabs.github.io/fastchess/)** · **[📦 PyPI](https://pypi.org/project/fastchess/)** · **[🐛 Issues](https://github.com/PeargentLabs/fastchess/issues)**

## Why fastchess?

| | fastchess | python-chess |
|---|---|---|
| **Core language** | C extension | Pure Python |
| **Move generation** | Native C - no Python overhead | Python loops |
| **ML tensor export** | Built-in `to_tensor()` → `(18, 8, 8)` NumPy array | Requires manual encoding |
| **UCI/SAN parsing** | C-level parsing | Python string ops |
| **Target use case** | Data pipelines, ML training, batch processing | General purpose |

If you're processing millions of chess positions for machine learning, fastchess gives you **C-speed move generation** with a clean Python API.

## Installation

```bash
pip install fastchess
```

Requires Python 3.9+ and NumPy. On most systems `pip` will handle compilation automatically.

## Quick Start

```python
import fastchess

board = fastchess.Board()                  # start position
board = fastchess.Board("rnbqkbnr/...")    # from FEN

# Move generation
moves = board.legal_moves_uci()           # ['e2e4', 'd2d4', ...]

# Apply moves (UCI or SAN notation)
board.push_uci("e2e4")
board.push_san("e5")

# Board state
print(board.fen())
print(board.turn)                         # fastchess.WHITE or fastchess.BLACK
print(board.is_check())

# ML tensor export (18 x 8 x 8 NumPy float32 array)
tensor = board.to_tensor(canonical=True)

# Board mirroring (useful for canonical ML input)
mirrored = board.mirror()
```

## ML / Tensor Export

Export any chess position as an `(18, 8, 8)` NumPy float32 array - ready for PyTorch or TensorFlow:

```python
import fastchess
import torch

board = fastchess.Board()
board.push_uci("e2e4")

tensor = board.to_tensor(canonical=True)  # (18, 8, 8) float32
x = torch.from_numpy(tensor).unsqueeze(0) # (1, 18, 8, 8) batch
```

The 18 planes encode: piece positions (6 per side), castling rights (4), en passant (1), and side to move (1).

## API

### `fastchess.Board(fen=None)`

| Method | Description |
|---|---|
| `copy()` | Return a copy of the board |
| `mirror()` | Return a vertically mirrored copy with swapped colors |
| `push_uci(uci)` | Apply a UCI move (e.g. `"e2e4"`) |
| `push_san(san)` | Apply a SAN move; returns the UCI string |
| `legal_moves_uci()` | List all legal moves as UCI strings |
| `piece_at(sq)` | `(piece_type, color)` at square index, or `None` |
| `is_check()` | True if the side to move is in check |
| `has_kingside_castling_rights(color)` | Castling rights query |
| `has_queenside_castling_rights(color)` | Castling rights query |
| `has_legal_en_passant()` | True if a legal en passant capture exists |
| `to_tensor(canonical=False)` | Board as `(18, 8, 8)` numpy array |
| `fen()` | Current FEN string |

### Constants

`WHITE`, `BLACK`, `PAWN`, `KNIGHT`, `BISHOP`, `ROOK`, `QUEEN`, `KING`

## PeargentLabs Ecosystem

fastchess is part of the **PeargentLabs** chess tooling ecosystem:

- **[fastchess](https://github.com/PeargentLabs/fastchess)** - High-performance Python chess library (C extension)
- **[OtterChess](https://github.com/PeargentLabs/OtterChess)** - Chess engine and analysis platform

## License

MIT - see [LICENSE](LICENSE) for details.

---

*Built with ♟ by [PeargentLabs](https://github.com/PeargentLabs)*
