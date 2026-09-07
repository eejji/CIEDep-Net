"""scripts/ 에서 src/ciedep 를 import 할 수 있게 경로를 등록한다."""

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
