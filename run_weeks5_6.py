"""Reproduce Weeks 5–6 experiments and the TA report without rerunning prior notebooks."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'src'))
from advanced_models import run
if __name__ == '__main__':
    run(ROOT)
    from validate_weeks5_6 import validate
    validate(ROOT)
    from build_weeks5_6_report import build
    build(ROOT)
