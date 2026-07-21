"""테스트 공통 설정."""

import sys
from pathlib import Path

import matplotlib

# 차트 생성 테스트가 GUI 백엔드를 띄우지 않도록, src.main을 import하기 전에 고정한다.
matplotlib.use("Agg")

# `python -m pytest` 외의 방식으로 실행해도 `src` 패키지를 찾을 수 있게 한다.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
