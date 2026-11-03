"""Run development-only calibration, drift analysis, validation and figures."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'src'))
from calibration_drift import run
if __name__ == '__main__':
    run(ROOT)
    from validate_weeks7_8 import validate
    print(validate(ROOT))
    from build_weeks7_8_report import build
    build(ROOT)
