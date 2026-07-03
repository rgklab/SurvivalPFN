from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
LIVE_TEXT_GLOBS = ["*.py", "*.yaml", "*.toml", "*.txt", "*.md"]
FORBIDDEN_PATTERNS = [
    "rasp" + "berry",
    "src." + "survivalpfn",
    "from " + "src",
    "from " + "constants",
    "from " + "data",
    "tab" + "icl",
    "Tab" + "ICL",
    "base" + "lines",
    "train_" + "parallel",
    "torch" + "run",
    "Distributed" + "DataParallel",
    "distributed " + "training",
    "multi-" + "GPU",
    "multiple " + "GPUs",
    "LOCAL_" + "RANK",
    "WORLD_" + "SIZE",
    "world_" + "size",
    "using_" + "dist",
    "init_" + "dist",
]


def iter_live_text_files():
    ignored_parts = {".git", ".pytest_cache", "output", "wandb", "__pycache__"}
    for pattern in LIVE_TEXT_GLOBS:
        for path in REPO_ROOT.rglob(pattern):
            if ignored_parts.intersection(path.parts):
                continue
            if path.name == "test_release_hygiene.py":
                continue
            yield path


def test_removed_release_patterns_are_not_in_live_text():
    hits = []
    for path in iter_live_text_files():
        text = path.read_text(encoding="utf-8")
        for pattern in FORBIDDEN_PATTERNS:
            if pattern in text:
                hits.append(f"{path.relative_to(REPO_ROOT)}: {pattern}")

    assert not hits, "Forbidden release-cleanup patterns found:\n" + "\n".join(hits)
