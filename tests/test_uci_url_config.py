import ast
import unittest
from pathlib import Path


UCI_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
ROOT = Path(__file__).resolve().parents[1]


def module_tree(path: str) -> ast.Module:
    return ast.parse((ROOT / path).read_text(encoding="utf-8"))


def assigned_string_constant(tree: ast.Module, name: str) -> str | None:
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                        return node.value.value
    return None


class UciUrlConfigTest(unittest.TestCase):
    def test_load_data_declares_official_uci_adult_data_url(self):
        tree = module_tree("src/load_data.py")

        self.assertEqual(assigned_string_constant(tree, "UCI_ADULT_DATA_URL"), UCI_URL)

    def test_main_uses_uci_url_as_default_data_source(self):
        source = (ROOT / "src/main.py").read_text(encoding="utf-8")

        self.assertIn('default=UCI_ADULT_DATA_URL', source)
        self.assertIn(UCI_URL, (ROOT / "src/load_data.py").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
