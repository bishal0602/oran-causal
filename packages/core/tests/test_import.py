import importlib


def test_member_imports() -> None:
    module = importlib.import_module("oran_causal.core")
    assert module.__name__ == "oran_causal.core"
