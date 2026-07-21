import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class MainDirectExecutionTest(unittest.TestCase):
    def test_main_module_runs_via_python_dash_m_from_project_root(self):
        """src/main.py는 패키지 내부 상대 import를 쓰므로 `python -m src.main`으로만 실행된다."""
        result = subprocess.run(
            [sys.executable, "-m", "src.main", "--help"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--data-path", result.stdout)


if __name__ == "__main__":
    unittest.main()
