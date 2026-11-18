"""Read-only historical account evidence. Replies are generated in generation.py."""
from functools import lru_cache
import json
from pathlib import Path
DATA=Path(__file__).resolve().parents[1]/'data/finance/replay.json'
@lru_cache(maxsize=1)
def evidence():
    if not DATA.exists():raise FileNotFoundError('Finance replay not prepared. Run finance/build_replay.py.')
    return json.loads(DATA.read_text(encoding='utf-8'))
